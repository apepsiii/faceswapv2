# jwt_config.py - Consistent JWT Secret Management

import os
from pathlib import Path
import secrets

JWT_SECRET_FILE = Path("jwt_secret.txt")

def get_or_create_jwt_secret():
    """Get existing JWT secret or create new one (consistent across restarts)"""
    
    if JWT_SECRET_FILE.exists():
        # Read existing secret
        with open(JWT_SECRET_FILE, 'r') as f:
            secret = f.read().strip()
        print(f"✅ Using existing JWT secret from {JWT_SECRET_FILE}")
        return secret
    else:
        # Create new secret
        secret = secrets.token_urlsafe(32)
        with open(JWT_SECRET_FILE, 'w') as f:
            f.write(secret)
        print(f"🔑 Created new JWT secret in {JWT_SECRET_FILE}")
        return secret

# Global JWT configuration
JWT_SECRET_KEY = get_or_create_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

print(f"🔐 JWT Secret Key: {JWT_SECRET_KEY[:20]}...")