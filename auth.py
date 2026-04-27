# EMPLOYEE_CHURN/auth.py
import json
import os
import hashlib
from functools import wraps
from flask import session, redirect, url_for

USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def _save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = _load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {"password": _hash_password(password)}
    _save_users(users)
    return True, "Registered successfully"

def login_user(username, password):
    users = _load_users()
    if username not in users:
        return False, "User not found"
    if users[username]["password"] != _hash_password(password):
        return False, "Incorrect password"
    return True, "Login successful"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated