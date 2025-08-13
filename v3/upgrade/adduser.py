# Python Script untuk Generate Real Password Hashes
# File: generate_site_users_hashes.py
# Usage: python generate_site_users_hashes.py

import hashlib
import secrets

def hash_password(password, salt=None):
    """Hash password using PBKDF2 (same method as in system)"""
    if salt is None:
        salt = secrets.token_hex(32)
    
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # 100,000 iterations
    )
    
    return password_hash.hex(), salt

def generate_sql_for_site_users():
    """Generate SQL commands dengan real password hashes"""
    
    site_codes = [
        'SLO', 'MGL', 'SDO', 'PLP', 'BRJ', 'LHT', 
        'KLK', 'MJN', 'KBN', 'CMG', 'PBN', 'BAT', 
        'SRG', 'BLO', 'BRU', 'BSD', 'CBT'
    ]
    
    print("-- SQL COMMANDS: Create 17 Site Users with REAL Password Hashes")
    print("-- Generated using PBKDF2 with 100,000 iterations")
    print("")
    
    for i, site_code in enumerate(site_codes, 1):
        username = site_code.lower()
        password = f"{username}123"  # slo123, mgl123, etc.
        
        # Generate real hash
        password_hash, salt = hash_password(password)
        
        print(f"-- {i}. {site_code} (username: {username}, password: {password})")
        print("INSERT INTO users (username, password_hash, salt, role, credit_balance, created_at)")
        print("VALUES (")
        print(f"    '{username}',")
        print(f"    '{password_hash}',")
        print(f"    '{salt}',")
        print("    'user',")
        print("    0,")
        print("    CURRENT_TIMESTAMP")
        print(");")
        print("")
    
    print("-- Verification Query")
    print("SELECT username, role, credit_balance, created_at")
    print("FROM users")
    print("WHERE role = 'user'")
    print("ORDER BY username;")

if __name__ == "__main__":
    generate_sql_for_site_users()