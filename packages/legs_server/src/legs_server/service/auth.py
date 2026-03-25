"""Authentication stub — mTLS / Token-based auth for gRPC interceptors.

This is a placeholder that always allows requests in development mode.
Replace with real mTLS validation or JWT/API-key checking for production.
"""

from __future__ import annotations

import logging
from typing import Optional

import grpc

logger = logging.getLogger(__name__)


class AuthInterceptor(grpc.ServerInterceptor):
    """Server-side gRPC interceptor for authentication.

    In development mode (token=None), all requests are allowed.
    When a token is configured, requests must carry it in the
    ``authorization`` metadata key.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token
        if token:
            logger.info("AuthInterceptor enabled (token-based)")
        else:
            logger.info("AuthInterceptor disabled (dev mode — all requests allowed)")

    def intercept_service(
        self,
        continuation: grpc.HandlerCallDetails,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        # In dev mode, pass through
        if self._token is None:
            return continuation(handler_call_details)

        # Check for token in metadata
        metadata = dict(handler_call_details.invocation_metadata or [])
        provided = metadata.get("authorization", "")

        if provided == f"Bearer {self._token}":
            return continuation(handler_call_details)

        logger.warning(
            "Auth rejected for method %s", handler_call_details.method,
        )
        return _denied_handler


def _deny(request, context):  # type: ignore[no-untyped-def]
    context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid or missing token")


_denied_handler = grpc.unary_unary_rpc_method_handler(_deny)
