"""Authentication — static API-Key + JWT, applied at the gateway boundary.

Two credential types (per the decision):
- **API-Key** (`X-API-Key` header): a static, configured key for trusted machine
  clients. Compared in constant time. Grants full scope (`*`).
- **JWT** (`Authorization: Bearer <token>`): issued by `POST /auth/token`,
  HS256-signed, with an expiry and a set of **scopes** for least-privilege access.

Security posture:
- `auth_enabled=True` by default; protected routes require a valid credential.
- If enabled with neither secret configured, authentication cannot succeed →
  protected routes deny (401). Secrets come from `.env` (gitignored).
- Constant-time API-key comparison; generic 401/403 messages (no oracle).

This module owns no routes; the gateway wires these dependencies onto routes and
the WS handshake.
"""
from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Header, HTTPException

from ..config import Settings

logger = logging.getLogger(__name__)

_FULL_SCOPE = "*"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: who, how, and with what scopes."""

    subject: str
    method: str # "api_key" | "jwt" | "anonymous"
    scopes: frozenset[str] = field(default_factory=frozenset)

    def has_scope(self, scope: str) -> bool:
        return _FULL_SCOPE in self.scopes or scope in self.scopes


# -- token helpers -----------------------------------------------------------
def create_access_token(settings: Settings, subject: str, scopes: list[str],
                        expires_minutes: int | None = None) -> str:
    """Mint an HS256 JWT with `sub`, `scopes`, `iat`, `exp`. Requires jwt_secret."""
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="token issuance not configured")
    minutes = settings.jwt_expiry_minutes if expires_minutes is None else expires_minutes
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": subject,
        "scopes": list(scopes),
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(settings: Settings, token: str) -> dict:
    """Decode/verify a JWT (signature + expiry). Raises 401 on any problem."""
    if not settings.jwt_secret:
        raise HTTPException(status_code=401, detail="invalid token")
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid token") from None


def _api_key_ok(settings: Settings, presented: str | None) -> bool:
    return bool(settings.api_key and presented and
                hmac.compare_digest(presented, settings.api_key))


def authenticate(settings: Settings, *, api_key: str | None,
                 authorization: str | None) -> Principal:
    """Resolve a Principal from raw credential values, or raise 401.

    Order: API-Key (full scope) → Bearer JWT (scoped). When auth is disabled,
    returns an anonymous full-scope principal."""
    if not settings.auth_enabled:
        return Principal(subject="anonymous", method="anonymous", scopes=frozenset({_FULL_SCOPE}))

    if api_key is not None and _api_key_ok(settings, api_key):
        return Principal(subject="api-key", method="api_key", scopes=frozenset({_FULL_SCOPE}))

    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        payload = decode_access_token(settings, token)
        scopes = frozenset(payload.get("scopes", []))
        return Principal(subject=str(payload.get("sub", "")), method="jwt", scopes=scopes)

    raise HTTPException(status_code=401, detail="authentication required",
                        headers={"WWW-Authenticate": "Bearer"})


def make_auth_dependencies(settings: Settings):
    """Build FastAPI dependencies bound to a specific Settings instance.

    Returns (require_principal, require_scope, require_api_key)."""

    async def require_principal(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> Principal:
        return authenticate(settings, api_key=x_api_key, authorization=authorization)

    def require_scope(scope: str):
        async def dep(
            x_api_key: str | None = Header(default=None, alias="X-API-Key"),
            authorization: str | None = Header(default=None),
        ) -> Principal:
            principal = authenticate(settings, api_key=x_api_key, authorization=authorization)
            if not principal.has_scope(scope):
                raise HTTPException(status_code=403, detail="insufficient scope")
            return principal
        return dep

    async def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> Principal:
        """For bootstrap-only routes (token issuance): API-Key required, not JWT."""
        if settings.auth_enabled and not _api_key_ok(settings, x_api_key):
            raise HTTPException(status_code=401, detail="API key required")
        return Principal(subject="api-key", method="api_key", scopes=frozenset({_FULL_SCOPE}))

    return require_principal, require_scope, require_api_key


def authenticate_ws(settings: Settings, *, api_key: str | None,
                    authorization: str | None, token_qs: str | None) -> Principal | None:
    """Authenticate a WebSocket handshake. Browsers can't set headers on a WS, so
    a `?token=<jwt>` query param is also accepted. Returns None on failure (the
    caller closes the socket with 1008) rather than raising."""
    if not settings.auth_enabled:
        return Principal(subject="anonymous", method="anonymous", scopes=frozenset({_FULL_SCOPE}))
    try:
        if token_qs:
            authorization = f"Bearer {token_qs}"
        return authenticate(settings, api_key=api_key, authorization=authorization)
    except HTTPException:
        return None
