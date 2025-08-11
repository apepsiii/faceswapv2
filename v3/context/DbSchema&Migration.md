# 🗃️ CONTEXT: DATABASE SCHEMA & MIGRATION

## **Current State:**
- Using SQLite database `face_swap.db`
- Existing tables: `users`, `face_swap_history`, `user_sessions`
- Basic authentication system working
- Single-role system (all users same level)

## **Target State:**
- Multi-role system (admin/user)
- Credit-based photo generation
- Transaction tracking
- Enhanced user management
- Analytics-ready schema

## **Migration Required:**

### **1. ALTER Existing Tables:**
```sql
-- Add role and credit to users table
ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user' CHECK(role IN ('admin', 'user'));
ALTER TABLE users ADD COLUMN credit_balance INTEGER DEFAULT 0;
```

### **2. Create New Tables:**
```sql
-- Transactions for payment tracking
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    order_id TEXT UNIQUE NOT NULL,
    amount INTEGER NOT NULL,
    credits_added INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Photos replace face_swap_history  
CREATE TABLE photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    photo_type TEXT NOT NULL CHECK(photo_type IN ('face_swap', 'ar_photo')),
    template_name TEXT,
    file_path TEXT NOT NULL,
    credits_used INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Settings for dynamic config
CREATE TABLE settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_name TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **3. Default Data:**
```sql
-- Default admin user
INSERT INTO users (username, password_hash, salt, role, credit_balance) 
VALUES ('admin', 'hashed_password', 'salt', 'admin', 999999);

-- Sample business users
INSERT INTO users (username, password_hash, salt, role, credit_balance) VALUES 
('cbt', 'hash', 'salt', 'user', 0),
('bsd', 'hash', 'salt', 'user', 0),
('slo', 'hash', 'salt', 'user', 0);

-- Default settings
INSERT INTO settings (key_name, value, description) VALUES 
('price_per_3_photos', '5000', 'Harga untuk 3 foto'),
('credits_per_payment', '3', 'Credit per pembayaran'),
('photos_per_session', '3', 'Foto per session');
```

## **Key Implementation Files:**
- `auth.py` - Enhanced authentication dengan role checking
- `main.py` - Database initialization dan migrations
- Migration script untuk existing data

## **Next Steps:**
1. Create migration script
2. Test dengan existing data
3. Update AuthService untuk role-based auth
4. Implement credit checking system