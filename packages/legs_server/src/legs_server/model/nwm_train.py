"""NWM training entry point — CLI-callable stub.

Usage:
    python -m legs_server.model.nwm_train --data-dir /data/rollouts --epochs 10

This is a placeholder that will be replaced with real training logic
(PyTorch / JAX) once the model architecture is finalized.
"""

from __future__ import annotations

import argparse
import logging
import time

logger = logging.getLogger(__name__)


def train(data_dir: str, epochs: int, output_dir: str, lr: float) -> str:
    """Run a training loop (stub).

    Returns:
        The model_id of the trained artifact.
    """
    logger.info(
        "Starting training [data_dir=%s, epochs=%d, output_dir=%s, lr=%s]",
        data_dir, epochs, output_dir, lr,
    )

    for epoch in range(1, epochs + 1):
        # Simulate training work
        time.sleep(0.01)
        loss = 1.0 / (epoch + 1)
        logger.info("Epoch %d/%d — loss=%.6f", epoch, epochs, loss)

    model_id = f"nwm-trained-ep{epochs}-{int(time.time())}"
    logger.info("Training complete. model_id=%s", model_id)
    return model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NWM training entry point")
    parser.add_argument("--data-dir", default="/data/rollouts", help="Path to training data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--output-dir", default="/data/models", help="Where to save checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    model_id = train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        output_dir=args.output_dir,
        lr=args.lr,
    )
    print(f"Trained model: {model_id}")


if __name__ == "__main__":
    main()
