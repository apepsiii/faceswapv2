#!/usr/bin/env python3
"""
Database Migration Script for Multi-Role Photobooth System
Migrates existing SQLite database to support:
- Role-based authentication (admin/user)
- Credit system
- Transaction tracking
- Enhanced photo metadata
"""

import sqlite3
import hashlib
import secrets
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigration:
    def __init__(self, db_path: Path = Path("face_swap.db")):
        self.db_path = db_path
        self.backup_path = Path(f"face_swap_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        
    def create_backup(self):
        """Create backup of existing database"""
        try:
            import shutil
            shutil.copy2(self.db_path, self.backup_path)
            logger.info(f"✅ Database backup created: {self.backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """Hash password with salt - same method as auth.py"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return password_hash.hex(), salt
    
    def check_existing_tables(self, conn):
        """Check what tables already exist"""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        existing_tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"📋 Existing tables: {existing_tables}")
        return existing_tables
    
    def migrate_users_table(self, conn):
        """Add role and credit_balance columns to users table"""
        try:
            # Check if columns already exist
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add role column if not exists
            if 'role' not in columns:
                conn.execute("""
                    ALTER TABLE users 
                    ADD COLUMN role TEXT DEFAULT 'user' 
                    CHECK(role IN ('admin', 'user'))
                """)
                logger.info("✅ Added 'role' column to users table")
            
            # Add credit_balance column if not exists
            if 'credit_balance' not in columns:
                conn.execute("""
                    ALTER TABLE users 
                    ADD COLUMN credit_balance INTEGER DEFAULT 0
                """)
                logger.info("✅ Added 'credit_balance' column to users table")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate users table: {e}")
            return False
    
    def create_transactions_table(self, conn):
        """Create transactions table for payment tracking"""
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    order_id TEXT UNIQUE NOT NULL,
                    amount INTEGER NOT NULL,
                    credits_added INTEGER NOT NULL,
                    payment_method TEXT DEFAULT 'qris',
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'settlement', 'failed', 'expired')),
                    midtrans_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            logger.info("✅ Created transactions table")
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_user ON transactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_status ON transactions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions(created_at)")
            logger.info("✅ Created transaction indexes")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create transactions table: {e}")
            return False
    
    def create_photos_table(self, conn):
        """Create photos table to replace face_swap_history"""
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    photo_type TEXT NOT NULL CHECK(photo_type IN ('face_swap', 'ar_photo')),
                    template_name TEXT,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    credits_used INTEGER DEFAULT 1,
                    processing_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            logger.info("✅ Created photos table")
            
            # Create indexes for dashboard queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photo_user ON photos(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photo_type ON photos(photo_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photo_date ON photos(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_user_type ON photos(user_id, photo_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_date_type ON photos(created_at, photo_type)")
            logger.info("✅ Created photo indexes")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create photos table: {e}")
            return False
    
    def create_settings_table(self, conn):
        """Create settings table for dynamic configuration"""
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_name TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    updated_by INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (updated_by) REFERENCES users (id)
                )
            """)
            logger.info("✅ Created settings table")
            
            # Insert default settings
            default_settings = [
                ('price_per_3_photos', '5000', 'Harga untuk 3 foto dalam rupiah'),
                ('credits_per_payment', '3', 'Jumlah credit yang didapat per pembayaran'),
                ('photos_per_session', '3', 'Maksimal foto per session sebelum bayar lagi'),
                ('currency', 'IDR', 'Mata uang yang digunakan'),
                ('business_name', 'Platinum Cineplex Photobooth', 'Nama bisnis'),
                ('admin_email', 'admin@platinumphotobooth.com', 'Email admin'),
                ('auto_backup_enabled', 'true', 'Enable automatic daily backup'),
                ('backup_retention_days', '30', 'Number of days to keep backups')
            ]
            
            for key_name, value, description in default_settings:
                conn.execute("""
                    INSERT OR IGNORE INTO settings (key_name, value, description)
                    VALUES (?, ?, ?)
                """, (key_name, value, description))
            
            logger.info("✅ Inserted default settings")
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create settings table: {e}")
            return False
    
    def migrate_existing_data(self, conn):
        """Migrate data from face_swap_history to photos table"""
        try:
            # Check if face_swap_history exists and has data
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='face_swap_history'
            """)
            
            if cursor.fetchone():
                # Get existing face swap records
                cursor = conn.execute("""
                    SELECT user_id, template_name, result_filename, created_at
                    FROM face_swap_history
                """)
                
                old_records = cursor.fetchall()
                
                if old_records:
                    logger.info(f"📦 Migrating {len(old_records)} records from face_swap_history")
                    
                    for record in old_records:
                        user_id, template_name, result_filename, created_at = record
                        
                        # Determine file path based on filename
                        file_path = f"static/results/{result_filename}"
                        
                        # Insert into photos table
                        conn.execute("""
                            INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (user_id, result_filename, 'face_swap', template_name, file_path, created_at))
                    
                    logger.info("✅ Successfully migrated face_swap_history data")
                    
                    # Optionally rename old table (don't delete for safety)
                    conn.execute("ALTER TABLE face_swap_history RENAME TO face_swap_history_old")
                    logger.info("✅ Renamed old table to face_swap_history_old")
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate existing data: {e}")
            return False
    
    def create_default_users(self, conn):
        """Create default admin user and sample business users"""
        try:
            # Create admin user
            admin_password_hash, admin_salt = self.hash_password("admin123")
            
            conn.execute("""
                INSERT OR IGNORE INTO users (username, password_hash, salt, role, credit_balance)
                VALUES (?, ?, ?, ?, ?)
            """, ("admin", admin_password_hash, admin_salt, "admin", 999999))
            
            logger.info("✅ Created default admin user (admin/admin123)")
            
            # Create sample business users
            business_users = [
                ("cbt", "cbt123"),
                ("bsd", "bsd123"), 
                ("slo", "slo123"),
                ("mgl", "mgl123"),
                ("sdo", "sdo123"),
                ("plp", "plp123")
            ]
            
            for username, password in business_users:
                password_hash, salt = self.hash_password(password)
                
                conn.execute("""
                    INSERT OR IGNORE INTO users (username, password_hash, salt, role, credit_balance)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, password_hash, salt, "user", 0))
            
            logger.info(f"✅ Created {len(business_users)} sample business users")
            
            # Show created users
            cursor = conn.execute("""
                SELECT username, role, credit_balance 
                FROM users 
                ORDER BY role DESC, username
            """)
            
            users = cursor.fetchall()
            logger.info("👥 Users in database:")
            for user in users:
                logger.info(f"   - {user[0]} ({user[1]}) - Credits: {user[2]}")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create default users: {e}")
            return False
    
    def create_user_directories(self):
        """Create user-specific directories for file organization"""
        try:
            base_dirs = [
                Path("static/results"),
                Path("static/ar_results")
            ]
            
            # Business users
            usernames = ["cbt", "bsd", "slo", "mgl", "sdo", "plp"]
            
            for base_dir in base_dirs:
                base_dir.mkdir(parents=True, exist_ok=True)
                
                for username in usernames:
                    user_dir = base_dir / username
                    user_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"📁 Created directory: {user_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create user directories: {e}")
            return False
    
    def optimize_database(self, conn):
        """Optimize database for performance"""
        try:
            # Apply SQLite optimizations
            optimizations = [
                "PRAGMA journal_mode = WAL",
                "PRAGMA synchronous = NORMAL",
                "PRAGMA cache_size = 10000",
                "PRAGMA temp_store = memory",
                "PRAGMA mmap_size = 268435456",
                "PRAGMA foreign_keys = ON"
            ]
            
            for pragma in optimizations:
                conn.execute(pragma)
                logger.info(f"✅ Applied: {pragma}")
            
            # Analyze tables for query optimizer
            conn.execute("ANALYZE")
            logger.info("✅ Analyzed database for query optimization")
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize database: {e}")
            return False
    
    def verify_migration(self, conn):
        """Verify migration completed successfully"""
        try:
            logger.info("🔍 Verifying migration...")
            
            # Check all required tables exist
            required_tables = ['users', 'transactions', 'photos', 'settings']
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            for table in required_tables:
                if table in existing_tables:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.error(f"❌ Table '{table}' missing")
                    return False
            
            # Check users table has role and credit_balance columns
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            
            required_columns = ['role', 'credit_balance']
            for column in required_columns:
                if column in columns:
                    logger.info(f"✅ Users table has '{column}' column")
                else:
                    logger.error(f"❌ Users table missing '{column}' column")
                    return False
            
            # Check admin user exists
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            if admin_count > 0:
                logger.info(f"✅ Found {admin_count} admin user(s)")
            else:
                logger.error("❌ No admin users found")
                return False
            
            # Check settings exist
            cursor = conn.execute("SELECT COUNT(*) FROM settings")
            settings_count = cursor.fetchone()[0]
            
            if settings_count > 0:
                logger.info(f"✅ Found {settings_count} settings")
            else:
                logger.error("❌ No settings found")
                return False
            
            logger.info("🎉 Migration verification completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False
    
    def run_migration(self):
        """Run complete database migration"""
        logger.info("🚀 Starting database migration...")
        
        # Create backup
        if not self.create_backup():
            logger.error("❌ Migration aborted - backup failed")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check existing tables
                existing_tables = self.check_existing_tables(conn)
                
                # Run migrations step by step
                steps = [
                    ("Migrate users table", self.migrate_users_table),
                    ("Create transactions table", self.create_transactions_table),
                    ("Create photos table", self.create_photos_table),
                    ("Create settings table", self.create_settings_table),
                    ("Migrate existing data", self.migrate_existing_data),
                    ("Create default users", self.create_default_users),
                    ("Optimize database", self.optimize_database),
                    ("Verify migration", self.verify_migration)
                ]
                
                for step_name, step_function in steps:
                    logger.info(f"🔄 {step_name}...")
                    
                    if not step_function(conn):
                        logger.error(f"❌ Migration failed at: {step_name}")
                        return False
                
                # Create user directories
                if not self.create_user_directories():
                    logger.warning("⚠️ Failed to create user directories (non-critical)")
                
                logger.info("🎉 Database migration completed successfully!")
                logger.info(f"📁 Backup saved at: {self.backup_path}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Migration failed with error: {e}")
            return False

def main():
    """Main migration function"""
    print("=" * 60)
    print("🗃️  PHOTOBOOTH DATABASE MIGRATION")
    print("=" * 60)
    
    migration = DatabaseMigration()
    
    # Confirm before proceeding
    print(f"📍 Database: {migration.db_path}")
    print(f"💾 Backup will be created: {migration.backup_path}")
    print()
    
    confirm = input("❓ Proceed with migration? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Migration cancelled by user")
        return False
    
    print()
    success = migration.run_migration()
    
    print("=" * 60)
    if success:
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("1. Test login with admin/admin123")
        print("2. Test login with business users (cbt/cbt123, etc)")
        print("3. Verify role-based access works")
        print("4. Test existing functionality still works")
    else:
        print("❌ MIGRATION FAILED!")
        print()
        print("Recovery steps:")
        print(f"1. Restore backup: cp {migration.backup_path} {migration.db_path}")
        print("2. Check error logs above")
        print("3. Fix issues and retry migration")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    main()