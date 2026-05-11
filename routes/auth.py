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
    Response:    { token, role, username, expires_at }
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


# ── POST /api/auth/register ───────────────────────────────────────────────────
@auth_bp.post("/register")
def register():
    """
    Register a new customer account.
    Body (JSON): { username, password }
    Response:    { token, role, username, expires_at }  (201)
    Errors:      409 if username taken, 400 if fields missing or too short
    """
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if len(username) < 3:
        return jsonify({"error": "username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400

    auth_svc = _services()["auth"]
    result = auth_svc.register(username, password)
    if result is None:
        return jsonify({"error": "Username already taken"}), 409

    return jsonify(result), 201
