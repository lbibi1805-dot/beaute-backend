"""
Auth routes — /api/auth/*
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request, current_app

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _services():
    return current_app.config["SERVICES"]


# ── POST /api/auth/login ──────────────────────────────────────────────────────
@auth_bp.post("/login")
def login():
    """
    Body (JSON): { username, password }
    Response:    { token, role, expires_at }
    """
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    auth_svc = _services()["auth"]
    result = auth_svc.login(username, password)
    if result is None:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify(result), 200
