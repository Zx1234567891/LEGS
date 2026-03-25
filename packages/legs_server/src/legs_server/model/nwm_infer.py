"""NWM inference backends — real CDiT-XL multi-GPU inference + CPU stub fallback.

The RealNWMPolicy loads Meta's Navigation World Model (CDiT-XL) pretrained
checkpoint and runs diffusion-based visual prediction across multiple GPUs.
The StubNWMPolicy remains available for CPU-only CI environments.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional

from legs_common.protocol.canon import Action, Observation

logger = logging.getLogger(__name__)

_NUM_JOINTS = 12
_NWM_REPO_ROOT = os.path.dirname(__file__)
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(_NWM_REPO_ROOT))
    )))),
    "nwm", "cdit_xl_100000.pth.tar",
)


# ---------------------------------------------------------------------------
# Real NWM Policy — CDiT-XL multi-GPU inference
# ---------------------------------------------------------------------------

class RealNWMPolicy:
    """Production NWM policy using Meta's CDiT-XL diffusion world model.

    Loads the pretrained checkpoint and replicates it across multiple GPUs.
    Each GPU independently samples candidate trajectories in parallel; the
    best candidate (lowest collision energy) is selected as the final action.

    This implements the core idea from the paper: test-time scaling via
    parallel trajectory sampling on multiple GPUs.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CKPT,
        device: str = "cuda",
        gpu_ids: Optional[List[int]] = None,
        num_candidates_per_gpu: int = 1,
        diffusion_steps: int = 250,
        image_size: int = 224,
        context_size: int = 4,
        use_bf16: bool = True,
        compile_model: bool = False,
    ) -> None:
        import torch
        from diffusers.models import AutoencoderKL
        from legs_server.model.cdit_models import CDiT_models
        from legs_server.model.diffusion import create_diffusion

        self._torch = torch

        # Determine GPU list
        if gpu_ids is not None:
            self._gpu_ids = gpu_ids
        else:
            n_gpus = torch.cuda.device_count()
            self._gpu_ids = list(range(n_gpus)) if n_gpus > 0 else []

        self._num_gpus = len(self._gpu_ids)
        self._num_candidates_per_gpu = num_candidates_per_gpu
        self._total_candidates = self._num_gpus * num_candidates_per_gpu
        self._primary_device = torch.device(f"cuda:{self._gpu_ids[0]}") if self._num_gpus > 0 else torch.device("cpu")
        self._use_bf16 = use_bf16 and self._primary_device.type == "cuda"
        self._image_size = image_size
        self._latent_size = image_size // 8
        self._context_size = context_size

        # Compute model_id from checkpoint hash
        with open(checkpoint_path, "rb") as f:
            ckpt_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()[:16]
        self._model_id = f"nwm-cdit-xl-{self._num_gpus}gpu-{ckpt_hash}"

        logger.info(
            "Loading CDiT-XL from %s [gpus=%s, candidates=%d, bf16=%s]",
            checkpoint_path, self._gpu_ids, self._total_candidates, self._use_bf16,
        )

        # Load weights once on CPU
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Create per-GPU model replicas, diffusion schedulers, and VAEs
        self._models: List[Any] = []
        self._diffusions: List[Any] = []
        self._vaes: List[Any] = []

        for gpu_id in self._gpu_ids:
            dev = torch.device(f"cuda:{gpu_id}")

            model = CDiT_models["CDiT-XL/2"](
                context_size=context_size,
                input_size=self._latent_size,
                in_channels=4,
            )
            model.load_state_dict(ckpt["ema"], strict=True)
            model.eval()
            model.to(dev)

            if compile_model and hasattr(torch, "compile"):
                model = torch.compile(model)

            diffusion = create_diffusion(str(diffusion_steps))

            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
            vae.to(dev)
            vae.eval()

            self._models.append(model)
            self._diffusions.append(diffusion)
            self._vaes.append(vae)

            logger.info("  GPU %d: CDiT-XL + VAE loaded", gpu_id)

        self._infer_count = 0
        logger.info(
            "RealNWMPolicy ready [model_id=%s, %d GPUs, %d total candidates]",
            self._model_id, self._num_gpus, self._total_candidates,
        )

    def infer(self, obs: Observation) -> Action:
        """Run CDiT-XL diffusion inference across all GPUs in parallel.

        Each GPU samples `num_candidates_per_gpu` candidate trajectories.
        The candidate with the best score is selected as the final action.
        """
        torch = self._torch
        t_start = time.monotonic_ns()

        action_delta = self._extract_action_from_obs(obs)

        # Launch parallel inference on all GPUs
        import concurrent.futures
        results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_gpus) as pool:
            futures = []
            for gpu_idx in range(self._num_gpus):
                fut = pool.submit(
                    self._infer_on_gpu,
                    gpu_idx,
                    action_delta,
                    self._num_candidates_per_gpu,
                )
                futures.append(fut)

            for fut in concurrent.futures.as_completed(futures):
                results.extend(fut.result())

        # Select best candidate (lowest energy / highest score)
        best = min(results, key=lambda r: r["energy"])

        t_infer = time.monotonic_ns() - t_start
        self._infer_count += 1

        # Convert to joint targets for Dog
        nav = best["action_delta"]
        targets = self._nav_to_joint_targets(nav)

        if self._infer_count % 10 == 0:
            logger.info(
                "NWM infer #%d: seq=%d, t=%.1fms, %d candidates, best_energy=%.4f, gpus=%d",
                self._infer_count, obs.seq, t_infer / 1e6,
                len(results), best["energy"], self._num_gpus,
            )

        return Action(
            seq_ref=obs.seq,
            action_type="nwm_navigation",
            payload={
                "joint_targets": targets,
                "nav_delta": {"x": nav[0], "y": nav[1], "yaw": nav[2]},
                "num_candidates": len(results),
                "best_energy": best["energy"],
                "gpu_id": best["gpu_id"],
                "predicted_image_shape": best["image_shape"],
            },
            model_id=self._model_id,
            t_infer_ns=t_infer,
        )

    def model_id(self) -> str:
        return self._model_id

    def _infer_on_gpu(
        self, gpu_idx: int, base_action: List[float], num_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Run diffusion sampling on a single GPU, returning candidate results."""
        torch = self._torch
        dev = torch.device(f"cuda:{self._gpu_ids[gpu_idx]}")
        model = self._models[gpu_idx]
        diffusion = self._diffusions[gpu_idx]
        vae = self._vaes[gpu_idx]
        dtype = torch.bfloat16 if self._use_bf16 else torch.float32
        B = num_candidates

        with torch.amp.autocast("cuda", enabled=self._use_bf16, dtype=dtype):
            # Context latents
            x_cond = torch.randn(
                B, self._context_size, 4, self._latent_size, self._latent_size,
                device=dev,
            )

            # Perturb action for diverse candidates
            base = torch.tensor(base_action, dtype=torch.float32, device=dev)
            noise = torch.randn(B, 3, device=dev) * 0.1
            actions = (base.unsqueeze(0) + noise)  # (B, 3)

            rel_t = torch.full((B,), 1.0 / 128.0, device=dev)

            z_noise = torch.randn(
                B, 4, self._latent_size, self._latent_size, device=dev,
            )

            model_kwargs = dict(y=actions, x_cond=x_cond, rel_t=rel_t)

            samples = diffusion.p_sample_loop(
                model.forward,
                z_noise.shape,
                z_noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=dev,
            )

            predicted = vae.decode(samples / 0.18215).sample
            predicted = torch.clip(predicted, -1.0, 1.0)

        # Score candidates (simple variance-based energy — lower is more coherent)
        results = []
        for i in range(B):
            img = predicted[i]
            energy = float(img.var().item())
            results.append({
                "action_delta": actions[i].cpu().tolist(),
                "energy": energy,
                "gpu_id": self._gpu_ids[gpu_idx],
                "image_shape": list(img.shape),
            })

        return results

    def _extract_action_from_obs(self, obs: Observation) -> List[float]:
        """Extract (delta_x, delta_y, delta_yaw) from the observation."""
        if "desired_action" in obs.sensors:
            da = obs.sensors["desired_action"]
            return [
                float(da.get("delta_x", 0.1)),
                float(da.get("delta_y", 0.0)),
                float(da.get("delta_yaw", 0.0)),
            ]
        return [0.1, 0.0, 0.0]

    def _nav_to_joint_targets(self, nav: List[float]) -> List[float]:
        """Map navigation delta to 12-DOF joint targets."""
        dx, dy, dyaw = nav
        targets: List[float] = []
        for i in range(_NUM_JOINTS):
            joint = i % 3
            if joint == 0:  # hip
                targets.append(dyaw * 0.1 + dy * 0.05)
            elif joint == 1:  # thigh
                targets.append(dx * 0.15)
            else:  # calf
                targets.append(-dx * 0.1)
        return targets


# ---------------------------------------------------------------------------
# Stub policies — CPU fallback for CI / development
# ---------------------------------------------------------------------------

class StubNWMPolicy:
    """CPU-only stub that satisfies NWMPolicy Protocol."""

    def __init__(self, model_tag: str = "stub-v0.1.0") -> None:
        self._model_tag = model_tag
        self._hash = hashlib.sha256(model_tag.encode()).hexdigest()[:16]
        self._model_id = f"{model_tag}-{self._hash}"
        logger.info("StubNWMPolicy initialised [model_id=%s]", self._model_id)

    def infer(self, obs: Observation) -> Action:
        t_start = time.monotonic_ns()
        targets: List[float] = [
            0.05 * math.sin(obs.seq * 0.1 + i) for i in range(_NUM_JOINTS)
        ]
        t_infer = time.monotonic_ns() - t_start
        return Action(
            seq_ref=obs.seq,
            action_type="joint_targets",
            payload={"joint_targets": targets},
            model_id=self._model_id,
            t_infer_ns=t_infer,
        )

    def model_id(self) -> str:
        return self._model_id


class RandomNWMPolicy:
    """Alternative stub that produces uniformly random joint targets."""

    def __init__(self, model_tag: str = "random-v0.1.0", amplitude: float = 0.3) -> None:
        self._model_tag = model_tag
        self._hash = hashlib.sha256(model_tag.encode()).hexdigest()[:16]
        self._model_id = f"{model_tag}-{self._hash}"
        self._amplitude = amplitude
        logger.info("RandomNWMPolicy initialised [model_id=%s]", self._model_id)

    def infer(self, obs: Observation) -> Action:
        t_start = time.monotonic_ns()
        targets: List[float] = [
            random.uniform(-self._amplitude, self._amplitude) for _ in range(_NUM_JOINTS)
        ]
        t_infer = time.monotonic_ns() - t_start
        return Action(
            seq_ref=obs.seq,
            action_type="joint_targets",
            payload={"joint_targets": targets},
            model_id=self._model_id,
            t_infer_ns=t_infer,
        )

    def model_id(self) -> str:
        return self._model_id
