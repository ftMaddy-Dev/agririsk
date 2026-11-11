"""API-key check for protected routes (basic shared-key protection, not per-user auth)."""
import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_header)) -> None:
    expected = os.getenv("AGRIRISK_API_KEY")
    if not expected:
        raise HTTPException(503, "API key is not configured on the server; this endpoint is disabled.")
    if key is None or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(401, "Missing or invalid API key.")
