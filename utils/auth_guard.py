"""
Auth guard decorator — protects admin-only Flask routes.

Usage:
    from utils.auth_guard import require_admin

    @admin_bp.get("/something")
    @require_admin
    def my_admin_view():
        ...
"""
from __future__ import annotations

from functools import wraps

from flask import current_app, request, jsonify

from enums.role import UserRole


def require_admin(fn):
    """Flask decorator that returns 401/403 unless the caller is an Admin."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.removeprefix("Bearer ").strip()
        auth_svc = current_app.config["SERVICES"]["auth"]
        role = auth_svc.verify(token)

        if role is None:
            return jsonify({"error": "Invalid or expired token"}), 401
        if role != UserRole.ADMIN:
            return jsonify({"error": "Admin access required"}), 403

        return fn(*args, **kwargs)
    return wrapper
