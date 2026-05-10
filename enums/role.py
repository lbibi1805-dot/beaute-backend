"""
UserRole enum — role-based access control.
Use this enum everywhere instead of raw role strings.
"""
from enum import Enum


class UserRole(str, Enum):
    ADMIN    = "admin"
    CUSTOMER = "customer"
