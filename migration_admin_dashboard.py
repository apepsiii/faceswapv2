# migration_admin_dashboard.py
"""
Database Migration Script for Admin Dashboard Analytics
Run this script to upgrade your existing database to support admin dashboard features
"""

import sqlite3
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdminDashboardMigration:
    def __init__(self, db_path: str = "face_swap.db"):
        self.db_path = Path(db_path)
        self.backup_path = Path(f"face_swap_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        
    def create_backup(self):
        """Create database backup before migration"""
        try:
            if self.db_path.exists():
                import shutil
                shutil.copy2(self.db_path, self.backup_path)
                logger.info(f"✅ Database backup created: {self.backup_path}")
            else:
                logger.info("📝 No existing database found, creating new one")
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            raise
    
    def check_existing_schema(self):
        """Check what tables and columns already exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                logger.info(f"📋 Existing tables: {existing_tables}")
                
                # Check users table structure
                if 'users' in existing_tables:
                    cursor = conn.execute("PRAGMA table_info(users)")
                    columns = [row[1] for row in cursor.fetchall()]
                    logger.info(f"👤 Users table columns: {columns}")
                    
                return existing_tables
                
        except Exception as e:
            logger.error(f"❌ Failed to check schema: {e}")
            return []
    
    def migrate_users_table(self):
        """Add role and credit_balance columns to users table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if columns already exist
                cursor = conn.execute("PRAGMA table_info(users)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'role' not in columns:
                    logger.info("🔧 Adding 'role' column to users table...")
                    conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user' CHECK(role IN ('admin', 'user'))")
                    
                if 'credit_balance' not in columns:
                    logger.info("🔧 Adding 'credit_balance' column to users table...")
                    conn.execute("ALTER TABLE users ADD COLUMN credit_balance INTEGER DEFAULT 0")
                
                conn.commit()
                logger.info("✅ Users table migration completed")
                
        except Exception as e:
            logger.error(f"❌ Failed to migrate users table: {e}")
            raise
    
    def create_transactions_table(self):
        """Create transactions table for payment tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        order_id TEXT UNIQUE NOT NULL,
                        amount INTEGER NOT NULL,
                        credits_added INTEGER NOT NULL,
                        status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'settlement', 'failed', 'expired')),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        settled_at TIMESTAMP,
                        payment_method TEXT DEFAULT 'qris',
                        notes TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_settled_at ON transactions(settled_at)")
                
                conn.commit()
                logger.info("✅ Transactions table created successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to create transactions table: {e}")
            raise
    
    def create_photos_table(self):
        """Create enhanced photos table to replace face_swap_history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT, -- JSON for additional data
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_user_id ON photos(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_type ON photos(photo_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_created_at ON photos(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_template ON photos(template_name)")
                
                conn.commit()
                logger.info("✅ Photos table created successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to create photos table: {e}")
            raise
    
    def create_settings_table(self):
        """Create settings table for dynamic configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key_name TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        value_type TEXT DEFAULT 'string' CHECK(value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
                        description TEXT,
                        category TEXT DEFAULT 'general',
                        is_public BOOLEAN DEFAULT FALSE,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by INTEGER,
                        FOREIGN KEY (updated_by) REFERENCES users (id)
                    )
                """)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)")
                
                conn.commit()
                logger.info("✅ Settings table created successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to create settings table: {e}")
            raise
    
    def migrate_existing_data(self):
        """Migrate data from face_swap_history to photos table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if face_swap_history exists and has data
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_swap_history'")
                if cursor.fetchone():
                    cursor = conn.execute("SELECT COUNT(*) FROM face_swap_history")
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        logger.info(f"🔄 Migrating {count} records from face_swap_history to photos...")
                        
                        # Migrate data
                        conn.execute("""
                            INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, created_at)
                            SELECT 
                                user_id,
                                result_filename,
                                'face_swap' as photo_type,
                                template_name,
                                '/static/results/' || result_filename as file_path,
                                created_at
                            FROM face_swap_history
                            WHERE NOT EXISTS (
                                SELECT 1 FROM photos p WHERE p.filename = face_swap_history.result_filename
                            )
                        """)
                        
                        migrated = conn.execute("SELECT changes()").fetchone()[0]
                        conn.commit()
                        
                        logger.info(f"✅ Migrated {migrated} records to photos table")
                
        except Exception as e:
            logger.error(f"❌ Failed to migrate existing data: {e}")
            raise
    
    def create_default_admin_user(self):
        """Create default admin user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if admin user already exists
                cursor = conn.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1")
                if cursor.fetchone():
                    logger.info("👤 Admin user already exists")
                    return
                
                # Create admin user
                username = "admin"
                password = "admin123"  # Change this in production!
                
                # Hash password
                salt = secrets.token_hex(32)
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                ).hex()
                
                conn.execute("""
                    INSERT INTO users (username, password_hash, salt, role, credit_balance)
                    VALUES (?, ?, ?, 'admin', 999999)
                """, (username, password_hash, salt))
                
                conn.commit()
                logger.info(f"✅ Default admin user created: {username} / {password}")
                logger.warning("⚠️  Please change the default admin password!")
                
        except Exception as e:
            logger.error(f"❌ Failed to create admin user: {e}")
            raise
    
    def create_sample_users(self):
        """Create sample business users"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                sample_users = ['cbt', 'bsd', 'slo', 'mgl', 'sdo', 'plp']
                
                for username in sample_users:
                    # Check if user exists
                    cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
                    if cursor.fetchone():
                        continue
                    
                    # Create user with default password
                    password = f"{username}123"
                    salt = secrets.token_hex(32)
                    password_hash = hashlib.pbkdf2_hmac(
                        'sha256',
                        password.encode('utf-8'),
                        salt.encode('utf-8'),
                        100000
                    ).hex()
                    
                    conn.execute("""
                        INSERT INTO users (username, password_hash, salt, role, credit_balance)
                        VALUES (?, ?, ?, 'user', 0)
                    """, (username, password_hash, salt))
                
                conn.commit()
                logger.info(f"✅ Sample users created: {', '.join(sample_users)}")
                
        except Exception as e:
            logger.error(f"❌ Failed to create sample users: {e}")
            raise
    
    def insert_default_settings(self):
        """Insert default system settings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                default_settings = [
                    ('price_per_3_photos', '5000', 'integer', 'Harga untuk 3 foto (IDR)', 'pricing'),
                    ('credits_per_payment', '3', 'integer', 'Credit yang didapat per pembayaran', 'pricing'),
                    ('photos_per_session', '3', 'integer', 'Maksimal foto per session', 'limits'),
                    ('auto_cleanup_days', '30', 'integer', 'Hapus foto otomatis setelah X hari', 'maintenance'),
                    ('system_maintenance', 'false', 'boolean', 'Mode maintenance system', 'system'),
                    ('max_file_size_mb', '10', 'integer', 'Maksimal ukuran file upload (MB)', 'limits'),
                    ('business_name', 'Platinum Cineplex', 'string', 'Nama bisnis', 'general'),
                    ('business_address', '', 'string', 'Alamat bisnis', 'general'),
                    ('contact_email', '', 'string', 'Email kontak', 'general'),
                    ('contact_phone', '', 'string', 'Nomor telepon kontak', 'general'),
                    ('enable_analytics', 'true', 'boolean', 'Aktifkan tracking analytics', 'analytics'),
                    ('session_timeout_minutes', '60', 'integer', 'Timeout session (menit)', 'security')
                ]
                
                for key_name, value, value_type, description, category in default_settings:
                    # Insert only if not exists
                    cursor = conn.execute("SELECT id FROM settings WHERE key_name = ?", (key_name,))
                    if not cursor.fetchone():
                        conn.execute("""
                            INSERT INTO settings (key_name, value, value_type, description, category)
                            VALUES (?, ?, ?, ?, ?)
                        """, (key_name, value, value_type, description, category))
                
                conn.commit()
                logger.info("✅ Default settings inserted")
                
        except Exception as e:
            logger.error(f"❌ Failed to insert default settings: {e}")
            raise
    
    def create_analytics_indexes(self):
        """Create additional indexes for analytics performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_users_role_active ON users(role, is_active)",
                    "CREATE INDEX IF NOT EXISTS idx_users_created_date ON users(DATE(created_at))",
                    "CREATE INDEX IF NOT EXISTS idx_photos_date_type ON photos(DATE(created_at), photo_type)",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_date_status ON transactions(DATE(settled_at), status)",
                    "CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount)",
                    "CREATE INDEX IF NOT EXISTS idx_photos_user_date ON photos(user_id, DATE(created_at))"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                conn.commit()
                logger.info("✅ Analytics indexes created")
                
        except Exception as e:
            logger.error(f"❌ Failed to create analytics indexes: {e}")
            raise
    
    def verify_migration(self):
        """Verify that migration was successful"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check all required tables exist
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['users', 'transactions', 'photos', 'settings']
                missing_tables = [table for table in required_tables if table not in tables]
                
                if missing_tables:
                    logger.error(f"❌ Missing tables: {missing_tables}")
                    return False
                
                # Check users table has role and credit_balance columns
                cursor = conn.execute("PRAGMA table_info(users)")
                user_columns = [row[1] for row in cursor.fetchall()]
                required_user_columns = ['role', 'credit_balance']
                missing_user_columns = [col for col in required_user_columns if col not in user_columns]
                
                if missing_user_columns:
                    logger.error(f"❌ Missing user columns: {missing_user_columns}")
                    return False
                
                # Check admin user exists
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                admin_count = cursor.fetchone()[0]
                
                if admin_count == 0:
                    logger.error("❌ No admin user found")
                    return False
                
                # Check settings exist
                cursor = conn.execute("SELECT COUNT(*) FROM settings")
                settings_count = cursor.fetchone()[0]
                
                if settings_count == 0:
                    logger.error("❌ No settings found")
                    return False
                
                logger.info("✅ Migration verification passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False
    
    def run_migration(self):
        """Run the complete migration process"""
        try:
            logger.info("🚀 Starting Admin Dashboard Migration...")
            
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Check existing schema
            existing_tables = self.check_existing_schema()
            
            # Step 3: Migrate users table
            self.migrate_users_table()
            
            # Step 4: Create new tables
            self.create_transactions_table()
            self.create_photos_table()
            self.create_settings_table()
            
            # Step 5: Migrate existing data
            self.migrate_existing_data()
            
            # Step 6: Create default users
            self.create_default_admin_user()
            self.create_sample_users()
            
            # Step 7: Insert default settings
            self.insert_default_settings()
            
            # Step 8: Create analytics indexes
            self.create_analytics_indexes()
            
            # Step 9: Verify migration
            if self.verify_migration():
                logger.info("🎉 Migration completed successfully!")
                self.print_summary()
            else:
                logger.error("❌ Migration verification failed!")
                raise Exception("Migration verification failed")
                
        except Exception as e:
            logger.error(f"💥 Migration failed: {e}")
            logger.info(f"📦 Database backup available at: {self.backup_path}")
            raise
    
    def print_summary(self):
        """Print migration summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count records
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
                user_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                admin_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM photos")
                photo_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM transactions")
                transaction_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM settings")
                settings_count = cursor.fetchone()[0]
                
                print("\n" + "="*60)
                print("📊 MIGRATION SUMMARY")
                print("="*60)
                print(f"👤 Total Users: {user_count}")
                print(f"👑 Admin Users: {admin_count}")
                print(f"📸 Total Photos: {photo_count}")
                print(f"💳 Total Transactions: {transaction_count}")
                print(f"⚙️  System Settings: {settings_count}")
                print("="*60)
                print("🔑 DEFAULT LOGIN CREDENTIALS:")
                print("   Username: admin")
                print("   Password: admin123")
                print("   ⚠️  CHANGE PASSWORD IMMEDIATELY!")
                print("="*60)
                print("📦 Backup Location:", self.backup_path)
                print("🌐 Admin Dashboard: http://localhost:5000/dashboard_admin")
                print("="*60)
                
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")

def main():
    """Main migration function"""
    print("🎯 AI Face Swap Studio - Admin Dashboard Migration")
    print("This will upgrade your database to support admin analytics dashboard")
    
    # Confirm migration
    response = input("\n⚠️  This will modify your database. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled")
        return
    
    try:
        migration = AdminDashboardMigration()
        migration.run_migration()
        
    except Exception as e:
        print(f"\n💥 Migration failed: {e}")
        print("📦 Check your database backup before retrying")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())