# Enhanced auth.py with role-based authentication
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

# Setup logging
logger = logging.getLogger(__name__)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"  # Enhanced with role
    initial_credits: int = 0  # Enhanced with initial credits

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    role: str  # Enhanced with role
    credit_balance: int  # Enhanced with credit balance
    created_at: str
    last_login: Optional[str] = None

# Configuration
JWT_SECRET_KEY = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Database setup
DB_PATH = Path("face_swap.db")

class EnhancedDatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database - enhanced version"""
        # Note: Migration script should have already created the enhanced schema
        # This method ensures the database exists and applies optimizations
        with sqlite3.connect(self.db_path) as conn:
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
                except Exception as e:
                    logger.warning(f"Failed to apply {pragma}: {e}")
            
            conn.commit()
            logger.info("Database optimizations applied")

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

class EnhancedAuthService:
    def __init__(self):
        self.db_manager = EnhancedDatabaseManager()
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def create_jwt_token(self, user_id: int, username: str, role: str) -> str:
        """Create JWT token for user - enhanced with role"""
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,  # Enhanced with role
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def register_user(self, user_data: UserCreate) -> Dict[str, Any]:
        """Register new user - enhanced with role and credits"""
        with self.db_manager.get_connection() as conn:
            # Check if username already exists
            cursor = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (user_data.username,)
            )
            
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username sudah digunakan"
                )
            
            # Validate password
            if len(user_data.password) < 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password minimal 4 karakter"
                )
            
            # Validate role
            if user_data.role not in ["admin", "user"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Role harus 'admin' atau 'user'"
                )
            
            # Hash password
            password_hash, salt = self.hash_password(user_data.password)
            
            # Insert new user with role and credits
            cursor = conn.execute("""
                INSERT INTO users (username, password_hash, salt, role, credit_balance)
                VALUES (?, ?, ?, ?, ?)
            """, (user_data.username, password_hash, salt, user_data.role, user_data.initial_credits))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"User created: {user_data.username} ({user_data.role}) with {user_data.initial_credits} credits")
            
            return {
                "success": True,
                "message": "User berhasil didaftarkan",
                "user_id": user_id,
                "username": user_data.username,
                "role": user_data.role
            }
    
    def login_user(self, login_data: UserLogin) -> Dict[str, Any]:
        """Login user and return token - enhanced with role and credits"""
        with self.db_manager.get_connection() as conn:
            # Get user data including role and credit_balance
            cursor = conn.execute("""
                SELECT id, username, password_hash, salt, role, credit_balance, is_active
                FROM users WHERE username = ?
            """, (login_data.username,))
            
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Username atau password salah"
                )
            
            user_id, username, password_hash, salt, role, credit_balance, is_active = user
            
            # Check if user is active
            if not is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Akun tidak aktif"
                )
            
            # Verify password
            if not self.verify_password(login_data.password, password_hash, salt):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Username atau password salah"
                )
            
            # Create JWT token with role
            token = self.create_jwt_token(user_id, username, role)
            
            # Update last login
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_id,))
            
            conn.commit()
            
            # Determine redirect URL based on role
            redirect_url = "/dashboard_admin" if role == "admin" else "/dashboard"
            
            logger.info(f"User logged in: {username} ({role}) → {redirect_url}")
            
            return {
                "success": True,
                "message": "Login berhasil",
                "token": token,
                "user": {
                    "id": user_id,
                    "username": username,
                    "role": role,
                    "credit_balance": credit_balance
                },
                "redirect_url": redirect_url  # Enhanced with role-based redirect
            }
    
    def get_user_by_token(self, token: str) -> Dict[str, Any]:
        """Get user data from token - enhanced with role and credits"""
        payload = self.verify_jwt_token(token)
        user_id = payload.get("user_id")
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, username, role, credit_balance, created_at, last_login
                FROM users WHERE id = ? AND is_active = TRUE
            """, (user_id,))
            
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User tidak ditemukan"
                )
            
            return {
                "id": user[0],
                "username": user[1],
                "role": user[2],  # Enhanced with role
                "credit_balance": user[3],  # Enhanced with credit balance
                "created_at": user[4],
                "last_login": user[5]
            }
    
    def update_user_credits(self, user_id: int, credits_change: int, reason: str = None) -> Dict[str, Any]:
        """Update user credit balance - new method"""
        with self.db_manager.get_connection() as conn:
            # Get current credit balance
            cursor = conn.execute(
                "SELECT username, credit_balance FROM users WHERE id = ?",
                (user_id,)
            )
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User tidak ditemukan"
                )
            
            username, current_credits = user
            new_credits = current_credits + credits_change
            
            # Prevent negative credits
            if new_credits < 0:
                new_credits = 0
            
            # Update credits
            conn.execute(
                "UPDATE users SET credit_balance = ? WHERE id = ?",
                (new_credits, user_id)
            )
            
            conn.commit()
            
            logger.info(f"Credits updated for {username}: {current_credits} → {new_credits} ({credits_change:+d})")
            
            return {
                "success": True,
                "message": f"Credits updated for {username}",
                "previous_credits": current_credits,
                "new_credits": new_credits,
                "change": credits_change
            }
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user statistics - new method"""
        with self.db_manager.get_connection() as conn:
            # Get user basic info
            cursor = conn.execute("""
                SELECT username, role, credit_balance, created_at, last_login
                FROM users WHERE id = ?
            """, (user_id,))
            
            user = cursor.fetchone()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User tidak ditemukan"
                )
            
            username, role, credit_balance, created_at, last_login = user
            
            # Get photo statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_photos,
                    COUNT(CASE WHEN photo_type = 'face_swap' THEN 1 END) as face_swap_count,
                    COUNT(CASE WHEN photo_type = 'ar_photo' THEN 1 END) as ar_photo_count,
                    MAX(created_at) as last_photo_date
                FROM photos WHERE user_id = ?
            """, (user_id,))
            
            photo_stats = cursor.fetchone()
            total_photos, face_swap_count, ar_photo_count, last_photo_date = photo_stats
            
            # Get transaction statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_transactions,
                    COALESCE(SUM(amount), 0) as total_spent,
                    COALESCE(SUM(credits_added), 0) as total_credits_purchased
                FROM transactions WHERE user_id = ? AND status = 'settlement'
            """, (user_id,))
            
            transaction_stats = cursor.fetchone()
            total_transactions, total_spent, total_credits_purchased = transaction_stats
            
            return {
                "user_info": {
                    "id": user_id,
                    "username": username,
                    "role": role,
                    "credit_balance": credit_balance,
                    "created_at": created_at,
                    "last_login": last_login
                },
                "photo_stats": {
                    "total_photos": total_photos or 0,
                    "face_swap_count": face_swap_count or 0,
                    "ar_photo_count": ar_photo_count or 0,
                    "last_photo_date": last_photo_date
                },
                "transaction_stats": {
                    "total_transactions": total_transactions or 0,
                    "total_spent": total_spent or 0,
                    "total_credits_purchased": total_credits_purchased or 0
                }
            }
    
    def logout_user(self, token: str) -> Dict[str, Any]:
        """Logout user and invalidate token"""
        # For JWT tokens, we rely on expiration
        # In production, you might want to maintain a blacklist
        return {
            "success": True,
            "message": "Logout berhasil"
        }

# Enhanced Authentication Middleware
security = HTTPBearer(auto_error=False)
auth_service = EnhancedAuthService()

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Dependency to get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    return auth_service.get_user_by_token(token)

async def admin_required(current_user: Dict = Depends(get_current_user)):
    """Dependency that requires admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def user_required(current_user: Dict = Depends(get_current_user)):
    """Dependency that requires user or admin role"""
    if current_user.get("role") not in ["admin", "user"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User access required"
        )
    return current_user

async def check_user_credits(current_user: Dict = Depends(get_current_user), required_credits: int = 1):
    """Dependency that checks if user has enough credits"""
    if current_user.get("role") == "admin":
        # Admins have unlimited credits
        return current_user
    
    if current_user.get("credit_balance", 0) < required_credits:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient credits. Required: {required_credits}, Available: {current_user.get('credit_balance', 0)}"
        )
    
    return current_user

# Utility functions for role checking
def is_admin(user: Dict) -> bool:
    """Check if user is admin"""
    return user.get("role") == "admin"

def is_user(user: Dict) -> bool:
    """Check if user is regular user"""
    return user.get("role") == "user"

def has_sufficient_credits(user: Dict, required_credits: int = 1) -> bool:
    """Check if user has enough credits"""
    if is_admin(user):
        return True  # Admins have unlimited credits
    return user.get("credit_balance", 0) >= required_credits

# Session management utilities
class SessionManager:
    """Enhanced session management for tracking user activity"""
    
    @staticmethod
    def create_session_data(user: Dict) -> Dict:
        """Create session data for frontend"""
        return {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "credit_balance": user["credit_balance"],
            "session_start": datetime.utcnow().isoformat(),
            "permissions": {
                "can_access_admin": is_admin(user),
                "can_generate_photos": has_sufficient_credits(user),
                "can_manage_users": is_admin(user),
                "can_view_analytics": is_admin(user)
            }
        }
    
    @staticmethod
    def update_session_credits(session_data: Dict, new_credit_balance: int) -> Dict:
        """Update credit balance in session data"""
        session_data["credit_balance"] = new_credit_balance
        session_data["permissions"]["can_generate_photos"] = new_credit_balance > 0 or session_data["role"] == "admin"
        return session_data

# Database connection validator
def validate_database_schema():
    """Validate that database has required enhanced schema"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Check required tables exist
            required_tables = ['users', 'transactions', 'photos', 'settings']
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            if missing_tables:
                logger.error(f"Missing required tables: {missing_tables}")
                return False
            
            # Check users table has required columns
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            
            required_columns = ['role', 'credit_balance']
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                logger.error(f"Users table missing columns: {missing_columns}")
                return False
            
            logger.info("Database schema validation passed")
            return True
            
    except Exception as e:
        logger.error(f"Database schema validation failed: {e}")
        return False

# Export enhanced auth service instance
enhanced_auth_service = EnhancedAuthService()

# Backwards compatibility - maintain existing interface
class AuthService(EnhancedAuthService):
    """Backwards compatibility wrapper"""
    pass

# For backwards compatibility with existing imports
if __name__ == "__main__":
    # Validate database schema
    if validate_database_schema():
        print("✅ Enhanced authentication service ready")
        print("✅ Database schema validated")
        print("\nFeatures:")
        print("- Role-based authentication (admin/user)")
        print("- Credit system integration")
        print("- Enhanced user statistics")
        print("- Session management")
        print("- Role-based middleware")
    else:
        print("❌ Database schema validation failed")
        print("Please run migration.py first")