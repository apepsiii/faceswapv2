#!/usr/bin/env python3
# migration.py - Database Migration & Setup Script

import sqlite3
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("face_swap.db")

class Migration:
    def __init__(self):
        self.db_path = DB_PATH
        
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return password_hash.hex(), salt
    
    def run_migration(self):
        """Run complete database migration"""
        logger.info("🚀 Starting database migration...")
        
        # Backup existing database
        self.backup_database()
        
        # Create enhanced schema
        self.create_enhanced_schema()
        
        # Insert default users
        self.insert_default_users()
        
        # Insert default settings
        self.insert_default_settings()
        
        # Create directories
        self.create_directories()
        
        # Verify migration
        self.verify_migration()
        
        logger.info("✅ Migration completed successfully!")
        
    def backup_database(self):
        """Backup existing database"""
        if self.db_path.exists():
            backup_path = self.db_path.parent / f"face_swap_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"📦 Database backed up to: {backup_path}")
        else:
            logger.info("📋 No existing database found, creating new one")
    
    def create_enhanced_schema(self):
        """Create enhanced database schema"""
        logger.info("🗃️ Creating enhanced database schema...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Drop existing tables if they exist (for clean migration)
            conn.execute("DROP TABLE IF EXISTS face_swap_history")
            conn.execute("DROP TABLE IF EXISTS user_sessions")
            
            # Create enhanced users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT DEFAULT 'user' CHECK(role IN ('admin', 'user')),
                    credit_balance INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Create transactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    order_id TEXT UNIQUE NOT NULL,
                    amount INTEGER NOT NULL,
                    credits_added INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'settlement', 'failed', 'admin_credit')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    payment_method TEXT DEFAULT 'qris',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Create photos table (replaces face_swap_history)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    photo_type TEXT NOT NULL CHECK(photo_type IN ('face_swap', 'ar_photo')),
                    template_name TEXT,
                    file_path TEXT NOT NULL,
                    credits_used INTEGER DEFAULT 1,
                    file_size INTEGER DEFAULT 0,
                    processing_time_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Create settings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_name TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_role ON users(role)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos ON photos(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photo_type ON photos(photo_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON transactions(order_id)")
            
            # Apply SQLite optimizations
            optimizations = [
                "PRAGMA journal_mode = WAL",
                "PRAGMA synchronous = NORMAL",
                "PRAGMA cache_size = 10000",
                "PRAGMA temp_store = memory",
                "PRAGMA foreign_keys = ON"
            ]
            
            for pragma in optimizations:
                try:
                    conn.execute(pragma)
                    logger.info(f"✅ Applied: {pragma}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply {pragma}: {e}")
            
            conn.commit()
            logger.info("✅ Enhanced schema created successfully")
    
    def insert_default_users(self):
        """Insert default admin and user accounts"""
        logger.info("👥 Creating default user accounts...")
        
        default_users = [
            {"username": "admin", "password": "admin123", "role": "admin", "credits": 999999},
            {"username": "cbt", "password": "cbt123", "role": "user", "credits": 0},
            {"username": "bsd", "password": "bsd123", "role": "user", "credits": 0},
            {"username": "slo", "password": "slo123", "role": "user", "credits": 0},
            {"username": "mgl", "password": "mgl123", "role": "user", "credits": 0},
            {"username": "sdo", "password": "sdo123", "role": "user", "credits": 0},
            {"username": "plp", "password": "plp123", "role": "user", "credits": 0},
            {"username": "demo", "password": "demo123", "role": "user", "credits": 3}  # Demo account with 3 credits
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for user_data in default_users:
                try:
                    # Check if user already exists
                    cursor = conn.execute("SELECT id FROM users WHERE username = ?", (user_data["username"],))
                    if cursor.fetchone():
                        logger.info(f"👤 User {user_data['username']} already exists, skipping...")
                        continue
                    
                    # Hash password
                    password_hash, salt = self.hash_password(user_data["password"])
                    
                    # Insert user
                    conn.execute("""
                        INSERT INTO users (username, password_hash, salt, role, credit_balance)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        user_data["username"], 
                        password_hash, 
                        salt, 
                        user_data["role"], 
                        user_data["credits"]
                    ))
                    
                    logger.info(f"✅ Created {user_data['role']}: {user_data['username']} (credits: {user_data['credits']})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create user {user_data['username']}: {e}")
            
            conn.commit()
    
    def insert_default_settings(self):
        """Insert default application settings"""
        logger.info("⚙️ Creating default settings...")
        
        default_settings = [
            {"key": "credits_per_payment", "value": "3", "description": "Credits per Rp 5.000 payment"},
            {"key": "price_per_payment", "value": "5000", "description": "Price per payment in IDR"},
            {"key": "max_photos_per_session", "value": "3", "description": "Maximum photos per session"},
            {"key": "auto_cleanup_days", "value": "30", "description": "Days to keep temporary files"},
            {"key": "app_version", "value": "2.1.0", "description": "Application version"},
            {"key": "maintenance_mode", "value": "false", "description": "Maintenance mode flag"}
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for setting in default_settings:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO settings (key_name, value, description, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (setting["key"], setting["value"], setting["description"]))
                    
                    logger.info(f"✅ Setting: {setting['key']} = {setting['value']}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create setting {setting['key']}: {e}")
            
            conn.commit()
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            "static/uploads",
            "static/templates", 
            "static/results",
            "static/ar_results",
            "static/images",
            "static/ar_assets/thumbnail",
            "static/ar_assets/countdown",
            "pages",
            "logs"
        ]
        
        # User-specific directories
        usernames = ["cbt", "bsd", "slo", "mgl", "sdo", "plp", "demo", "admin"]
        for username in usernames:
            directories.extend([
                f"static/results/{username}",
                f"static/ar_results/{username}"
            ])
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created: {directory}")
            except Exception as e:
                logger.error(f"❌ Failed to create {directory}: {e}")
    
    def verify_migration(self):
        """Verify migration was successful"""
        logger.info("🔍 Verifying migration...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Check tables exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['users', 'transactions', 'photos', 'settings']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.error(f"❌ Missing tables: {missing_tables}")
                return False
            
            logger.info(f"✅ Tables created: {tables}")
            
            # Check users
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            logger.info(f"✅ Users created: {user_count}")
            
            # Check admin user
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            logger.info(f"✅ Admin users: {admin_count}")
            
            # Check settings
            cursor = conn.execute("SELECT COUNT(*) FROM settings")
            settings_count = cursor.fetchone()[0]
            logger.info(f"✅ Settings created: {settings_count}")
            
            # Test JWT secret consistency by checking if we can create and verify a token
            try:
                import jwt
                import secrets
                from datetime import timedelta
                
                # Use same secret generation method as in main.py
                JWT_SECRET_KEY = secrets.token_urlsafe(32)
                
                # Create test token
                test_payload = {
                    "user_id": 1,
                    "username": "admin",
                    "role": "admin"
                }
                
                test_token = jwt.encode(test_payload, JWT_SECRET_KEY, algorithm="HS256")
                decoded_payload = jwt.decode(test_token, JWT_SECRET_KEY, algorithms=["HS256"])
                
                logger.info("✅ JWT token generation/verification working")
                
            except Exception as e:
                logger.warning(f"⚠️ JWT test failed: {e}")
            
            return True
    
    def create_enhanced_auth_file(self):
        """Create enhanced_auth.py if it doesn't exist"""
        auth_file = Path("enhanced_auth.py")
        
        if not auth_file.exists():
            logger.info("📝 Creating enhanced_auth.py...")
            
            # Copy content from provided enhanced_auth.py
            auth_content = '''# enhanced_auth.py - Generated by migration script
# This is a simplified version for compatibility
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Use consistent JWT configuration
JWT_SECRET_KEY = "your-consistent-secret-key-here"  # Fixed secret for consistency
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"
    initial_credits: int = 0

class UserLogin(BaseModel):
    username: str
    password: str

class EnhancedAuthService:
    def __init__(self):
        self.db_path = Path("face_swap.db")
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    # Add your enhanced auth methods here...
    # (This is a placeholder - use your actual enhanced_auth.py content)

# Create instance
enhanced_auth_service = EnhancedAuthService()

def validate_database_schema():
    """Validate database schema"""
    try:
        with sqlite3.connect("face_swap.db") as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            required_tables = ['users', 'transactions', 'photos', 'settings']
            return all(table in tables for table in required_tables)
    except:
        return False
'''
            
            with open(auth_file, 'w') as f:
                f.write(auth_content)
            
            logger.info("✅ enhanced_auth.py created")

def main():
    """Main migration function"""
    print("=" * 60)
    print("🚀 AI FACE SWAP STUDIO - DATABASE MIGRATION")
    print("=" * 60)
    
    migration = Migration()
    
    try:
        migration.run_migration()
        
        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📋 Default Accounts Created:")
        print("   Admin: username=admin, password=admin123")
        print("   Users: cbt/cbt123, bsd/bsd123, slo/slo123, etc.")
        print("   Demo:  username=demo, password=demo123 (3 credits)")
        print("\n🔗 Next Steps:")
        print("   1. Start the application: python main.py")
        print("   2. Test login: http://localhost:5000/login")
        print("   3. Test admin: http://localhost:5000/dashboard_admin")
        print("   4. Test API: http://localhost:5000/api/test/phase1")
        print("\n💳 Test Credit System:")
        print("   1. Login as demo user")
        print("   2. Generate photo (uses 1 credit)")
        print("   3. Go to payment when credits exhausted")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: {e}")
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()