# update_database.py - Script untuk update database dan create admin
import sqlite3
import hashlib
import secrets
from pathlib import Path

def update_database():
    """Update database schema dan create admin user"""
    db_path = Path("face_swap.db")
    
    if not db_path.exists():
        print("❌ Database face_swap.db tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan aplikasi sekali untuk membuat database.")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            print("🔧 Updating database schema...")
            
            # 1. Check and add role column
            try:
                cursor = conn.execute("SELECT role FROM users LIMIT 1")
                print("✓ Role column already exists")
            except sqlite3.OperationalError:
                print("  Adding role column...")
                conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
                conn.execute("UPDATE users SET role = 'user' WHERE role IS NULL OR role = ''")
                print("✓ Role column added")
            
            # 2. Check and add credit_balance column
            try:
                cursor = conn.execute("SELECT credit_balance FROM users LIMIT 1")
                print("✓ Credit_balance column already exists")
            except sqlite3.OperationalError:
                print("  Adding credit_balance column...")
                conn.execute("ALTER TABLE users ADD COLUMN credit_balance INTEGER DEFAULT 0")
                conn.execute("UPDATE users SET credit_balance = 0 WHERE credit_balance IS NULL")
                print("✓ Credit_balance column added")
            
            # 3. Check if admin user exists
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                print("  Creating admin user...")
                
                # Create admin user
                admin_username = "admin"
                admin_password = "admin123"
                
                # Hash password using same method as AuthService
                salt = secrets.token_hex(32)
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    admin_password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                ).hex()
                
                # Insert admin user
                conn.execute("""
                    INSERT INTO users (username, password_hash, salt, role, credit_balance, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (admin_username, password_hash, salt, "admin", 999999, True))
                
                print(f"✓ Admin user created: {admin_username} / {admin_password}")
            else:
                # Update existing admin to have admin role
                conn.execute("UPDATE users SET role = 'admin', credit_balance = 999999 WHERE username = 'admin'")
                print("✓ Admin user already exists and updated")
            
            # 4. Create additional tables for future features
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    order_id TEXT UNIQUE NOT NULL,
                    amount INTEGER NOT NULL,
                    credits_added INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            print("✓ Transactions table created/verified")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    photo_type TEXT NOT NULL CHECK(photo_type IN ('face_swap', 'ar_photo')),
                    template_name TEXT,
                    file_path TEXT NOT NULL,
                    credits_used INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            print("✓ Photos table created/verified")
            
            # 5. Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos ON photos(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions ON transactions(user_id)")
            print("✓ Database indexes created")
            
            conn.commit()
            print("\n🎉 Database update completed successfully!")
            
            # Show current users
            cursor = conn.execute("SELECT username, role, credit_balance FROM users ORDER BY role DESC, username")
            users = cursor.fetchall()
            
            print("\n📋 Current users in database:")
            for user in users:
                role = user[1] if user[1] else 'user'
                credits = user[2] if user[2] else 0
                print(f"  - {user[0]} (role: {role}, credits: {credits})")
                
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        return False
    
    return True

def check_database_status():
    """Check current database status"""
    db_path = Path("face_swap.db")
    
    if not db_path.exists():
        print("❌ Database tidak ditemukan")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            print("📊 Database Status:")
            
            # Check tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"  Tables: {', '.join(tables)}")
            
            # Check users table structure
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"  Users columns: {', '.join(columns)}")
            
            # Check users
            cursor = conn.execute("SELECT username, role, credit_balance FROM users")
            users = cursor.fetchall()
            print(f"  Total users: {len(users)}")
            
            for user in users:
                role = user[1] if len(user) > 1 and user[1] else 'user'
                credits = user[2] if len(user) > 2 and user[2] else 0
                print(f"    - {user[0]} (role: {role}, credits: {credits})")
                
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    print("🚀 Database Update Script")
    print("=" * 40)
    
    # Check current status
    check_database_status()
    
    print("\n" + "=" * 40)
    
    # Ask for confirmation
    confirm = input("Do you want to update the database? (y/N): ").lower().strip()
    
    if confirm in ['y', 'yes']:
        if update_database():
            print("\n✅ Update completed! You can now login as admin:")
            print("   Username: admin")
            print("   Password: admin123")
            print("\n🔗 Admin dashboard will be available at: /dashboard_admin")
        else:
            print("\n❌ Update failed!")
    else:
        print("Update cancelled.")