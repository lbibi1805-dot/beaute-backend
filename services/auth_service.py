"""
AuthService — demo-grade authentication.

Token format: base64(role + ":" + AUTH_SECRET)
No JWT library needed — sufficient for a local demo.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta

import config
from enums.role import UserRole

# Fixed credential table (defined in config, no external DB)
_CREDENTIALS: dict[str, tuple[str, UserRole]] = {
    config.ADMIN_USERNAME:    (config.ADMIN_PASSWORD,    UserRole.ADMIN),
    config.CUSTOMER_USERNAME: (config.CUSTOMER_PASSWORD, UserRole.CUSTOMER),
}

_TOKEN_TTL_HOURS = 8


def _encode(role: UserRole) -> str:
    raw = f"{role.value}:{config.AUTH_SECRET}"
    return base64.b64encode(raw.encode()).decode()


def _decode(token: str) -> UserRole | None:
    try:
        raw = base64.b64decode(token.encode()).decode()
        role_str, secret = raw.split(":", 1)
        if secret != config.AUTH_SECRET:
            return None
        return UserRole(role_str)
    except Exception:
        return None


class AuthService:
    def login(self, username: str, password: str) -> dict | None:
        """
        Validate credentials and return auth info dict, or None if invalid.
        Response shape: {token, role, expires_at}
        """
        entry = _CREDENTIALS.get(username)
        if entry is None:
            return None
        expected_password, role = entry
        if password != expected_password:
            return None

        expires_at = datetime.now(timezone.utc) + timedelta(hours=_TOKEN_TTL_HOURS)
        return {
            "token":      _encode(role),
            "role":       role.value,
            "expires_at": expires_at.isoformat(),
        }

    def verify(self, token: str) -> UserRole | None:
        """Decode and return the role from a token, or None if invalid."""
        return _decode(token)
