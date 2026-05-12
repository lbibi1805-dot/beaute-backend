"""
AuthService — demo-grade authentication + in-memory registration.

Token format: base64(role + ":" + username + ":" + AUTH_SECRET)
No JWT library needed — sufficient for a local demo.
"""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone, timedelta

import config
from enums.role import UserRole

# Fixed credential table (defined in config, no external DB)
_FIXED_CREDENTIALS: dict[str, tuple[str, UserRole]] = {
    config.ADMIN_USERNAME:    (config.ADMIN_PASSWORD,    UserRole.ADMIN),
    config.CUSTOMER_USERNAME: (config.CUSTOMER_PASSWORD, UserRole.CUSTOMER),
}

_TOKEN_TTL_HOURS = 8


def _encode(role: UserRole, username: str) -> str:
    raw = f"{role.value}:{username}:{config.AUTH_SECRET}"
    return base64.b64encode(raw.encode()).decode()


def _decode(token: str) -> tuple[UserRole, str] | None:
    """Return (role, username) or None if the token is invalid."""
    try:
        raw = base64.b64decode(token.encode()).decode()
        parts = raw.split(":", 2)
        if len(parts) != 3 or parts[2] != config.AUTH_SECRET:
            return None
        return UserRole(parts[0]), parts[1]
    except Exception:
        return None


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


class AuthService:
    def __init__(self) -> None:
        # Registered users (in-memory, demo-grade — not persisted across restarts)
        self._registered: dict[str, tuple[str, UserRole]] = {}

    def login(self, username: str, password: str) -> dict | None:
        """
        Validate credentials and return auth info dict, or None if invalid.
        Response shape: {token, role, username, expires_at}
        """
        # Check fixed credentials first
        entry = _FIXED_CREDENTIALS.get(username)
        if entry is not None:
            expected_password, role = entry
            if password != expected_password:
                return None
        else:
            # Check dynamically registered users
            reg = self._registered.get(username)
            if reg is None:
                return None
            pw_hash, role = reg
            if _hash_pw(password) != pw_hash:
                return None

        expires_at = datetime.now(timezone.utc) + timedelta(hours=_TOKEN_TTL_HOURS)
        return {
            "token":      _encode(role, username),
            "role":       role.value,
            "username":   username,
            "expires_at": expires_at.isoformat(),
        }

    def register(self, username: str, password: str) -> dict | None:
        """
        Register a new customer account. Returns auth info dict, or None if
        the username is already taken (fixed or registered).
        """
        if username in _FIXED_CREDENTIALS or username in self._registered:
            return None  # conflict

        self._registered[username] = (_hash_pw(password), UserRole.CUSTOMER)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=_TOKEN_TTL_HOURS)
        return {
            "token":      _encode(UserRole.CUSTOMER, username),
            "role":       UserRole.CUSTOMER.value,
            "username":   username,
            "expires_at": expires_at.isoformat(),
        }

    def verify(self, token: str) -> UserRole | None:
        """Decode and return the role from a token, or None if invalid."""
        result = _decode(token)
        return result[0] if result else None

    def get_username(self, token: str) -> str | None:
        """Decode and return the username from a token, or None if invalid."""
        result = _decode(token)
        return result[1] if result else None
