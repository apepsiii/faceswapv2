# main.py - Part 1: Imports & Configuration

from fastapi import FastAPI, UploadFile, Query, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter
from pathlib import Path
from fastapi import FastAPI, Depends, Request
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time, uuid
from midtrans_config import core_api
from PIL import Image
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import logging
import aiofiles
import asyncio
from contextlib import asynccontextmanager
import uuid
import mimetypes
import requests
from pathlib import Path
import shutil

from jwt_config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS

# Import authentication modules
import sqlite3
import hashlib
import secrets
from datetime import timedelta
import jwt
from pydantic import BaseModel
import threading
from insightface.app import FaceAnalysis
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    UPLOAD_DIR = Path("static/uploads")
    TEMPLATE_DIR = Path("static/templates")
    RESULT_DIR = Path("static/results")
    FRAME_DIR = Path("static/images")
    PAGES_DIR = Path("pages")
    
    # AR Photo directories
    AR_ASSETS_DIR = Path("static/ar_assets")
    COUNTDOWN_DIR = Path("static/ar_assets/countdown")
    AR_RESULTS_DIR = Path("static/ar_results")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/bmp'}
    
    # Face detection parameters
    DET_SIZE = (640, 640)
    CTX_ID = 0
    
    # JWT Configuration - USE CONSISTENT SECRET
    JWT_SECRET_KEY = JWT_SECRET_KEY
    JWT_ALGORITHM = JWT_ALGORITHM
    JWT_EXPIRATION_HOURS = JWT_EXPIRATION_HOURS

# Also update the BasicAuthService class to use consistent JWT secret:

class BasicAuthService:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def create_jwt_token(self, user_id: int, username: str, role: str = 'user') -> str:
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),  # Use imported constant
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)  # Use imported constants
        return token
    
    def verify_jwt_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])  # Use imported constants
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

# Enhanced Authentication Imports - FIXED
try:
    from enhanced_auth import (
        EnhancedAuthService, 
        enhanced_auth_service,
        admin_required, 
        user_required, 
        check_user_credits,
        UserCreate,
        UserLogin,
        is_admin,
        has_sufficient_credits,
        SessionManager,
        get_current_user,
        validate_database_schema
    )
    ENHANCED_AUTH_AVAILABLE = True
    logger.info("✅ Enhanced authentication imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced auth not available: {e}")
    logger.warning("Falling back to basic auth - run migration.py first")
    ENHANCED_AUTH_AVAILABLE = False

# Pydantic models for authentication (fallback)
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# Configuration
class Config:
    UPLOAD_DIR = Path("static/uploads")
    TEMPLATE_DIR = Path("static/templates")
    RESULT_DIR = Path("static/results")
    FRAME_DIR = Path("static/images")
    PAGES_DIR = Path("pages")
    
    # AR Photo directories
    AR_ASSETS_DIR = Path("static/ar_assets")
    COUNTDOWN_DIR = Path("static/ar_assets/countdown")
    AR_RESULTS_DIR = Path("static/ar_results")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/bmp'}
    
    # Face detection parameters
    DET_SIZE = (640, 640)
    CTX_ID = 0
    
    # JWT Configuration
    JWT_SECRET_KEY = secrets.token_urlsafe(32)
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24

# Global variables
face_app = None
swapper = None
DB_PATH = Path("face_swap.db")

# Fallback Basic Auth Classes (if enhanced auth not available)
class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            # Basic schema for fallback
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    credit_balance INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
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
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    photo_type TEXT NOT NULL,
                    template_name TEXT,
                    file_path TEXT NOT NULL,
                    credits_used INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos ON photos(user_id)")
            conn.commit()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

class BasicAuthService:
    def __init__(self):
        self.db_manager = DatabaseManager()
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
    
    def create_jwt_token(self, user_id: int, username: str, role: str = 'user') -> str:
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
        return token
    
    def verify_jwt_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
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
    
    def get_user_by_token(self, token: str) -> dict:
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
                "role": user[2] or 'user',
                "credit_balance": user[3] or 0,
                "created_at": user[4],
                "last_login": user[5]
            }

    def login_user(self, login_data: UserLogin) -> dict:
        with self.db_manager.get_connection() as conn:
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
            
            if not is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Akun tidak aktif"
                )
            
            if not self.verify_password(login_data.password, password_hash, salt):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Username atau password salah"
                )
            
            role = role or 'user'
            credit_balance = credit_balance or 0
            
            token = self.create_jwt_token(user_id, username, role)
            
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_id,))
            
            conn.commit()
            
            redirect_url = "/dashboard_admin" if role == "admin" else "/dashboard"
            
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
                "redirect_url": redirect_url
            }

# Initialize authentication service
if ENHANCED_AUTH_AVAILABLE:
    auth_service = enhanced_auth_service
    logger.info("✅ Using Enhanced Authentication Service")
else:
    auth_service = BasicAuthService()
    logger.warning("⚠️ Using Basic Authentication Service - run migration.py for enhanced features")

# Security and dependencies
security = HTTPBearer(auto_error=False)

# Authentication dependencies - FIXED
if ENHANCED_AUTH_AVAILABLE:
    # Use enhanced auth dependencies
    pass  # They're already imported
else:
    # Create basic auth dependencies
    async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        token = credentials.credentials
        return auth_service.get_user_by_token(token)
    
    async def admin_required(current_user = Depends(get_current_user)):
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return current_user
    
    async def user_required(current_user = Depends(get_current_user)):
        if current_user.get("role") not in ["admin", "user"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User access required"
            )
        return current_user
    
    async def check_user_credits(current_user = Depends(get_current_user)):
        # Get fresh credit balance
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT credit_balance FROM users WHERE id = ?", 
                (current_user["id"],)
            )
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=401, detail="User not found")
            
            credit_balance = result[0] or 0
        
        if credit_balance < 1:
            raise HTTPException(
                status_code=402,
                detail={
                    "message": "Kredit habis! Silakan lakukan pembayaran untuk melanjutkan.",
                    "credits_remaining": 0,
                    "redirect_to": "/payment"
                }
            )
        
        current_user["credit_balance"] = credit_balance
        return current_user
    
# main.py - Part 2: Lifespan, App Setup & Core Functions

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with migration verification"""
    global face_app, swapper
    
    try:
        logger.info("🚀 Starting enhanced application...")
        
        # Verify database schema if enhanced auth is available
        if ENHANCED_AUTH_AVAILABLE:
            try:
                if not validate_database_schema():
                    logger.error("❌ Database schema validation failed!")
                    logger.error("Please run migration.py first")
                else:
                    logger.info("✅ Database schema validated")
            except Exception as e:
                logger.warning(f"⚠️ Schema validation failed: {e}")
                logger.warning("Continuing with basic auth...")
        
        # Initialize face analysis models
        logger.info("Initializing face analysis model...")
        face_app = FaceAnalysis(name='buffalo_l')
        face_app.prepare(ctx_id=Config.CTX_ID, det_size=Config.DET_SIZE)
        
        logger.info("Loading face swapper model...")
        swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', 
            download=False, 
            download_zip=False
        )
        
        # Create all directories
        directories = [
            Config.UPLOAD_DIR, Config.TEMPLATE_DIR, Config.RESULT_DIR, 
            Config.FRAME_DIR, Config.PAGES_DIR, Config.AR_ASSETS_DIR, 
            Config.COUNTDOWN_DIR, Config.AR_RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create user-specific directories if enhanced auth available
        if ENHANCED_AUTH_AVAILABLE:
            usernames = ["cbt", "bsd", "slo", "mgl", "sdo", "plp"]
            for username in usernames:
                (Config.RESULT_DIR / username).mkdir(parents=True, exist_ok=True)
                (Config.AR_RESULTS_DIR / username).mkdir(parents=True, exist_ok=True)
            logger.info("✅ User-specific directories created")
        
        logger.info("✅ Application initialization complete!")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Application shutdown")

# ESP32 Configuration
ESP32_IP = "10.65.124.246"
lampu_status = "off"

# FastAPI App Creation
app = FastAPI(
    title="AI Face Swap Studio",
    description="Advanced Face Swapping API with Authentication and AR Photo",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom exceptions
class FaceSwapError(Exception):
    pass

class ValidationError(Exception):
    pass

# Utility functions
def validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise ValidationError("Filename tidak boleh kosong")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Ekstensi file tidak didukung. Gunakan: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
    
    mime_type = mimetypes.guess_type(file.filename)[0]
    if mime_type not in Config.ALLOWED_MIME_TYPES:
        raise ValidationError(f"Tipe file tidak didukung: {mime_type}")

def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_ext = Path(original_filename).suffix.lower()
    
    safe_filename = f"{prefix}{timestamp}_{unique_id}{file_ext}"
    return safe_filename

async def save_uploaded_file(file: UploadFile, save_path: Path) -> Path:
    try:
        async with aiofiles.open(save_path, 'wb') as f:
            content = await file.read()
            
            if len(content) > Config.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File terlalu besar. Maksimal {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            await f.write(content)
        
        logger.info(f"File saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error saving file {save_path}: {e}")
        if save_path.exists():
            save_path.unlink()
        raise

def detect_faces(image_path: Path) -> List:
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar: {image_path}")
        
        faces = face_app.get(img)
        logger.info(f"Detected {len(faces)} faces in {image_path}")
        return faces
    
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        raise FaceSwapError(f"Error dalam deteksi wajah: {e}")

def swap_faces(src_path: Path, dst_path: Path, output_path: Path) -> Path:
    try:
        img_src = cv2.imread(str(src_path))
        img_dst = cv2.imread(str(dst_path))
        
        if img_src is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar sumber: {src_path}")
        if img_dst is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar template: {dst_path}")
        
        faces_src = detect_faces(src_path)
        faces_dst = detect_faces(dst_path)
        
        if len(faces_src) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada gambar sumber")
        if len(faces_dst) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada template")
        
        face_src = faces_src[0]
        face_dst = faces_dst[0]
        
        result = swapper.get(img_dst.copy(), face_dst, face_src, paste_back=True)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), result)
        
        if not success:
            raise FaceSwapError("Gagal menyimpan hasil face swap")
        
        logger.info(f"Face swap completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Face swap error: {e}")
        if isinstance(e, FaceSwapError):
            raise
        raise FaceSwapError(f"Error dalam proses face swap: {e}")

def overlay_frame(base_image_path: Path, frame_path: Path, output_path: Path) -> Path:
    try:
        if not frame_path.exists():
            logger.warning(f"Frame file not found: {frame_path}")
            return base_image_path
        
        base_img = cv2.imread(str(base_image_path), cv2.IMREAD_UNCHANGED)
        frame_img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        
        if base_img is None:
            raise FaceSwapError(f"Tidak dapat membaca gambar dasar: {base_image_path}")
        if frame_img is None:
            logger.warning(f"Cannot read frame image: {frame_path}")
            return base_image_path
        
        frame_img = cv2.resize(frame_img, (base_img.shape[1], base_img.shape[0]))
        
        if frame_img.shape[2] == 4:
            alpha_mask = frame_img[:, :, 3] / 255.0
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            base_img_rgb = base_img[:, :, :3]
            frame_img_rgb = frame_img[:, :, :3]
            
            result = (1 - alpha_mask) * base_img_rgb + alpha_mask * frame_img_rgb
            result = result.astype(np.uint8)
        else:
            result = cv2.addWeighted(base_img, 0.7, frame_img, 0.3, 0)
        
        success = cv2.imwrite(str(output_path), result)
        if not success:
            raise FaceSwapError("Gagal menyimpan hasil overlay frame")
        
        logger.info(f"Frame overlay completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Frame overlay error: {e}")
        return base_image_path

def apply_ar_overlay(base_image_path: Path, overlay_path: Path, output_path: Path) -> Path:
    """Apply AR overlay to captured photo"""
    try:
        if not overlay_path.exists():
            logger.warning(f"Overlay file not found: {overlay_path}")
            shutil.copy2(base_image_path, output_path)
            return output_path
        
        base_img = cv2.imread(str(base_image_path), cv2.IMREAD_UNCHANGED)
        overlay_img = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
        
        if base_img is None:
            raise FaceSwapError(f"Cannot read base image: {base_image_path}")
        if overlay_img is None:
            logger.warning(f"Cannot read overlay image: {overlay_path}")
            shutil.copy2(base_image_path, output_path)
            return output_path
        
        overlay_img = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))
        
        if overlay_img.shape[2] == 4:
            alpha_mask = overlay_img[:, :, 3] / 255.0
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)
            
            base_img_rgb = base_img[:, :, :3] if base_img.shape[2] >= 3 else base_img
            overlay_img_rgb = overlay_img[:, :, :3]
            
            result = (1 - alpha_mask) * base_img_rgb + alpha_mask * overlay_img_rgb
            result = result.astype(np.uint8)
        else:
            result = cv2.addWeighted(base_img, 0.7, overlay_img, 0.3, 0)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), result)
        
        if not success:
            raise FaceSwapError("Failed to save AR overlay result")
        
        logger.info(f"AR overlay applied: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"AR overlay error: {e}")
        shutil.copy2(base_image_path, output_path)
        return output_path

def serve_html_page(page_name: str) -> HTMLResponse:
    """Serve HTML page from pages directory"""
    try:
        page_path = Config.PAGES_DIR / f"{page_name}.html"
        if page_path.exists():
            with open(page_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html><head><title>Page Not Found</title></head>
            <body>
            <h1>Page: {page_name}</h1>
            <p>File {page_name}.html tidak ditemukan di folder pages/</p>
            <p>Silakan letakkan file HTML di folder pages/</p>
            <p><a href="/login">Go to Login</a></p>
            </body></html>
            """)
    except Exception as e:
        logger.error(f"Error serving page {page_name}: {e}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html><head><title>Error</title></head>
        <body>
        <h1>Error</h1>
        <p>Terjadi kesalahan saat memuat halaman {page_name}</p>
        <p>Error: {str(e)}</p>
        </body></html>
        """)

# Credit management functions
async def deduct_user_credit(user_id: int, credits_to_deduct: int = 1) -> int:
    """Deduct credits from user and return new balance"""
    with auth_service.db_manager.get_connection() as conn:
        # Get current balance first
        cursor = conn.execute("SELECT credit_balance FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        current_balance = result[0] or 0
        
        if current_balance < credits_to_deduct:
            raise HTTPException(
                status_code=402,
                detail={
                    "message": f"Kredit tidak mencukupi. Saldo: {current_balance}",
                    "credits_remaining": current_balance,
                    "credits_needed": credits_to_deduct
                }
            )
        
        # Deduct credit
        new_balance = current_balance - credits_to_deduct
        conn.execute(
            "UPDATE users SET credit_balance = ? WHERE id = ?",
            (new_balance, user_id)
        )
        conn.commit()
        
        logger.info(f"Credit deducted: -{credits_to_deduct} for user_id {user_id}, new balance: {new_balance}")
        return new_balance
    
# main.py - Part 3: API Routes & Endpoints

# ===== FRONTEND ROUTES =====

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><title>AI Face Swap Studio</title>
    <meta http-equiv="refresh" content="0; url=/login">
    </head>
    <body>
    <p>Redirecting to <a href="/login">Login Page</a>...</p>
    </body></html>
    """)

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return serve_html_page("login")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    return serve_html_page("dashboard")

@app.get("/dashboard_admin", response_class=HTMLResponse)
async def admin_dashboard_page():
    return serve_html_page("dashboard_admin")

@app.get("/character", response_class=HTMLResponse)
async def character_page():
    return serve_html_page("character")

@app.get("/camera", response_class=HTMLResponse)
async def camera_page():
    global lampu_status
    lampu_status = "on"
    return serve_html_page("camera")

@app.get("/ar-character", response_class=HTMLResponse)
async def ar_character_page():
    return serve_html_page("ar_character")

@app.get("/ar_camera", response_class=HTMLResponse)
async def ar_camera_page():
    return serve_html_page("ar_camera")

@app.get("/result", response_class=HTMLResponse)
async def result_page():
    return serve_html_page("result")

@app.get("/manipulasi", response_class=HTMLResponse)
async def manipulasi_page():
    return serve_html_page("manipulasi")

@app.get("/ar_manipulasi", response_class=HTMLResponse)
async def ar_manipulasi_page():
    return serve_html_page("ar_manipulasi")

@app.get("/payment", response_class=HTMLResponse)
async def payment_page():
    return serve_html_page("payment")

@app.get("/ar_payment", response_class=HTMLResponse)
async def ar_payment_page():
    return serve_html_page("ar_payment")

@app.get("/ar-result", response_class=HTMLResponse)
async def ar_result_page():
    return serve_html_page("ar_result")

# ===== ESP32 & RELAY ROUTES =====

@app.get("/relay")
async def relay_control(state: str):
    if state not in ["on", "off"]:
        return JSONResponse({"success": False, "message": "Invalid state"}, status_code=400)
    
    try:
        resp = requests.get(f"http://{ESP32_IP}/relay?state={state}", timeout=2)
        if resp.status_code == 200:
            return {"success": True}
        else:
            return JSONResponse({"success": False, "message": "ESP error"}, status_code=500)
    except:
        return JSONResponse({"success": False, "message": "ESP unreachable"}, status_code=500)

@app.get("/status")
async def get_status():
    return {"status": lampu_status}

@app.get("/reset-lampu")
async def reset_lampu():
    global lampu_status
    lampu_status = "off"
    return {"success": True}

# ===== QRIS PAYMENT ROUTES =====

@app.get("/api/qris/token")
async def generate_qris_token(current_user = Depends(get_current_user)):
    """Enhanced QRIS generation with user tracking"""
    try:
        order_id = f"ORDER-{uuid.uuid4().hex[:12]}"
        amount = 5000  # Rp 5,000
        credits_to_add = 3  # 3 credits per payment
        
        # Record transaction in database
        with auth_service.db_manager.get_connection() as conn:
            conn.execute("""
                INSERT INTO transactions (user_id, order_id, amount, credits_added, status)
                VALUES (?, ?, ?, ?, ?)
            """, (current_user["id"], order_id, amount, credits_to_add, "pending"))
            conn.commit()
        
        # Generate QRIS payload
        payload = {
            "payment_type": "qris",
            "transaction_details": {
                "order_id": order_id,
                "gross_amount": amount,
            },
            "qris": {
                "acquirer": "gopay"
            },
            "custom_field1": str(current_user["id"]),
            "custom_field2": "credit_purchase"
        }
        
        result = core_api.charge(payload)
        logger.info(f"QRIS generated for user {current_user['username']}, order_id: {order_id}")
        
        # Extract QR URL
        actions = result.get("actions", [])
        qris_url = next((a["url"] for a in actions if a.get("name") == "generate-qr-code"), None)
        
        if not qris_url:
            logger.error(f"QRIS URL not found for order_id: {order_id}")
            
            # Cleanup failed transaction
            with auth_service.db_manager.get_connection() as conn:
                conn.execute("DELETE FROM transactions WHERE order_id = ?", (order_id,))
                conn.commit()
            
            raise HTTPException(status_code=400, detail="QRIS URL tidak ditemukan")
        
        return {
            "success": True, 
            "qris_url": qris_url, 
            "order_id": order_id,
            "amount": amount,
            "credits_to_add": credits_to_add,
            "user_id": current_user["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QRIS generation failed for user {current_user['username']}: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup failed transaction
        try:
            with auth_service.db_manager.get_connection() as conn:
                conn.execute("DELETE FROM transactions WHERE order_id = ?", (order_id,))
                conn.commit()
        except:
            pass
        
        raise HTTPException(status_code=500, detail="Gagal membuat QRIS. Silakan coba lagi.")

@app.get("/api/qris/status")
async def check_qris_status(order_id: str):
    """Enhanced status check with auto-credit addition"""
    try:
        # Check Midtrans status
        status = core_api.transactions.status(order_id)
        logger.info(f"QRIS status check for {order_id}: {status.get('transaction_status')}")
        
        # If payment successful, add credits
        if status.get("transaction_status") == "settlement":
            with auth_service.db_manager.get_connection() as conn:
                # Get transaction details
                cursor = conn.execute("""
                    SELECT user_id, credits_added, status 
                    FROM transactions 
                    WHERE order_id = ?
                """, (order_id,))
                
                transaction = cursor.fetchone()
                
                if transaction and transaction[2] == "pending":
                    user_id, credits_to_add, current_status = transaction
                    
                    # Add credits to user
                    conn.execute("""
                        UPDATE users 
                        SET credit_balance = credit_balance + ? 
                        WHERE id = ?
                    """, (credits_to_add, user_id))
                    
                    # Update transaction status
                    conn.execute("""
                        UPDATE transactions 
                        SET status = 'settlement', settled_at = CURRENT_TIMESTAMP 
                        WHERE order_id = ?
                    """, (order_id,))
                    
                    conn.commit()
                    
                    # Get user info for logging
                    cursor = conn.execute("SELECT username, credit_balance FROM users WHERE id = ?", (user_id,))
                    user_info = cursor.fetchone()
                    
                    if user_info:
                        logger.info(f"Credits added: {credits_to_add} to user {user_info[0]}, new balance: {user_info[1]}")
                        status["credits_added"] = credits_to_add
                        status["new_credit_balance"] = user_info[1]
        
        return status
        
    except Exception as e:
        logger.error(f"QRIS status check failed for {order_id}: {e}")
        return {"error": str(e), "transaction_status": "failure"}

@app.post("/api/qris/webhook")
async def qris_webhook(request: Request):
    """Webhook for automatic payment notification"""
    try:
        body = await request.json()
        order_id = body.get("order_id")
        transaction_status = body.get("transaction_status")
        
        logger.info(f"Webhook received: {order_id} - {transaction_status}")
        
        if transaction_status == "settlement":
            await check_qris_status(order_id)
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return {"status": "error"}

# ===== AUTHENTICATION ENDPOINTS =====

@app.post("/api/register")
async def register(user_data: UserCreate):
    """Enhanced register endpoint with role support"""
    if ENHANCED_AUTH_AVAILABLE:
        return auth_service.register_user(user_data)
    else:
        # Basic registration for fallback
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM users WHERE username = ?", (user_data.username,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Username sudah digunakan")
            
            if len(user_data.password) < 4:
                raise HTTPException(status_code=400, detail="Password minimal 4 karakter")
            
            password_hash, salt = auth_service.hash_password(user_data.password)
            
            cursor = conn.execute("""
                INSERT INTO users (username, password_hash, salt, role, credit_balance)
                VALUES (?, ?, ?, ?, ?)
            """, (user_data.username, password_hash, salt, 'user', 0))
            
            conn.commit()
            
            return {
                "success": True,
                "message": "User berhasil didaftarkan",
                "user_id": cursor.lastrowid
            }

@app.post("/api/login")
async def login(login_data: UserLogin):
    """Enhanced login endpoint with role-based redirect"""
    result = auth_service.login_user(login_data)
    
    if ENHANCED_AUTH_AVAILABLE:
        logger.info(f"Enhanced login: {result['user']['username']} ({result['user']['role']}) → {result['redirect_url']}")
    else:
        logger.info(f"Basic login: {result['user']['username']} → {result['redirect_url']}")
    
    return result

@app.post("/api/logout")
async def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Enhanced logout endpoint"""
    if not credentials:
        return {"success": True, "message": "Logout berhasil"}
    
    return {"success": True, "message": "Logout berhasil"}

@app.get("/api/me")
async def get_me(current_user = Depends(get_current_user)):
    """Enhanced user info endpoint with role and credits"""
    return {
        "success": True,
        "user": current_user
    }

# ===== CREDIT MANAGEMENT ENDPOINTS =====

@app.get("/api/user/credits")
async def get_user_credits(current_user = Depends(get_current_user)):
    """Get current user credit balance"""
    
    # Get fresh balance from database
    with auth_service.db_manager.get_connection() as conn:
        cursor = conn.execute("""
            SELECT 
                u.credit_balance,
                COUNT(p.id) as total_photos_taken,
                COALESCE(SUM(t.credits_added), 0) as total_credits_purchased
            FROM users u
            LEFT JOIN photos p ON u.id = p.user_id
            LEFT JOIN transactions t ON u.id = t.user_id AND t.status = 'settlement'
            WHERE u.id = ?
            GROUP BY u.id
        """, (current_user["id"],))
        
        result = cursor.fetchone()
        
        if result:
            credit_balance, total_photos, total_purchased = result
        else:
            credit_balance, total_photos, total_purchased = current_user.get("credit_balance", 0), 0, 0
    
    return {
        "success": True,
        "user_id": current_user["id"],
        "username": current_user["username"],
        "credit_balance": credit_balance,
        "total_photos_taken": total_photos,
        "total_credits_purchased": total_purchased,
        "credits_per_payment": 3,
        "price_per_payment": 5000
    }

# ===== PHOTO GENERATION ENDPOINTS =====

@app.post("/api/swap")
async def swap_faces_api(
    template_name: str = Form(...),
    webcam: UploadFile = File(...),
    source: Optional[UploadFile] = File(None),
    apply_frame: bool = Form(True),
    current_user = Depends(check_user_credits)
):
    """Enhanced face swap with credit system and user folders"""
    temp_files = []
    
    try:
        # Validate files
        validate_file(webcam)
        if source:
            validate_file(source)
        
        # Check template exists
        template_path = Config.TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' tidak ditemukan")
        
        # Generate user-specific filename
        username = current_user["username"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        result_filename = f"{username}_{timestamp}_{unique_id}.png"
        
        # Create user-specific folder
        user_result_dir = Config.RESULT_DIR / username
        user_result_dir.mkdir(parents=True, exist_ok=True)
        result_path = user_result_dir / result_filename
        
        # Save uploaded files temporarily
        webcam_filename = generate_unique_filename(webcam.filename, "webcam_")
        webcam_path = Config.UPLOAD_DIR / webcam_filename
        await save_uploaded_file(webcam, webcam_path)
        temp_files.append(webcam_path)
        
        source_path = webcam_path
        if source:
            source_filename = generate_unique_filename(source.filename, "source_")
            source_path = Config.UPLOAD_DIR / source_filename
            await save_uploaded_file(source, source_path)
            temp_files.append(source_path)
        
        logger.info(f"Starting face swap for user {username}: {source_path} -> {template_path}")
        
        # Process face swap
        swap_result_path = swap_faces(source_path, template_path, result_path)
        
        # Apply frame overlay if requested
        final_result_path = swap_result_path
        if apply_frame:
            frame_path = Config.FRAME_DIR / "frame1.png"
            final_result_path = overlay_frame(swap_result_path, frame_path, result_path)
        
        # Deduct credit and record photo in database
        with auth_service.db_manager.get_connection() as conn:
            # Deduct credit
            new_credit_balance = await deduct_user_credit(current_user["id"], 1)
            
            # Record photo
            conn.execute("""
                INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, credits_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                current_user["id"], 
                result_filename, 
                "face_swap", 
                template_name, 
                str(final_result_path),
                1
            ))
            
            conn.commit()
        
        # Detect faces for response
        faces_detected = {
            "source": len(detect_faces(source_path)),
            "template": len(detect_faces(template_path))
        }
        
        response_data = {
            "success": True,
            "message": "Face swap berhasil dilakukan",
            "data": {
                "result_url": f"/static/results/{username}/{result_filename}",
                "result_filename": result_filename,
                "template_used": template_name,
                "faces_detected": faces_detected,
                "frame_applied": apply_frame,
                "processing_time": datetime.now().isoformat(),
                "credits_remaining": new_credit_balance,
                "user_folder": username,
                "file_size": final_result_path.stat().st_size if final_result_path.exists() else 0
            }
        }
        
        logger.info(f"Face swap successful for {username}: {final_result_path}, credits remaining: {new_credit_balance}")
        return JSONResponse(response_data)
    
    except HTTPException:
        raise
    except ValidationError as e:
        logger.warning(f"Validation error for user {current_user.get('username', 'unknown')}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FaceSwapError as e:
        logger.error(f"Face swap error for user {current_user.get('username', 'unknown')}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in face swap for user {current_user.get('username', 'unknown')}: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada server")
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")

@app.post("/api/ar/photo")
async def create_ar_photo(
    photo: UploadFile = File(...),
    overlay_name: str = Form("frame1.png"),
    current_user = Depends(check_user_credits)
):
    """Enhanced AR photo with credit system"""
    temp_files = []
    
    try:
        # Validate uploaded photo
        validate_file(photo)
        
        # Generate user-specific filename
        username = current_user["username"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        result_filename = f"{username}_{timestamp}_{unique_id}_ar.png"
        
        # Create user-specific AR folder
        user_ar_dir = Config.AR_RESULTS_DIR / username
        user_ar_dir.mkdir(parents=True, exist_ok=True)
        result_path = user_ar_dir / result_filename
        
        # Save uploaded photo temporarily
        photo_filename = generate_unique_filename(photo.filename, "ar_capture_")
        photo_path = Config.UPLOAD_DIR / photo_filename
        await save_uploaded_file(photo, photo_path)
        temp_files.append(photo_path)
        
        # Define overlay path
        overlay_path = Config.FRAME_DIR / overlay_name
        
        logger.info(f"Starting AR photo for user {username}: {photo_path} + {overlay_name}")
        
        # Apply AR overlay
        final_result_path = apply_ar_overlay(photo_path, overlay_path, result_path)
        
        # Deduct credit and record photo
        with auth_service.db_manager.get_connection() as conn:
            # Deduct credit
            new_credit_balance = await deduct_user_credit(current_user["id"], 1)
            
            # Record AR photo
            conn.execute("""
                INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, credits_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                current_user["id"],
                result_filename,
                "ar_photo", 
                f"AR_OVERLAY_{overlay_name}",
                str(final_result_path),
                1
            ))
            
            conn.commit()
        
        response_data = {
            "success": True,
            "message": "AR photo berhasil dibuat",
            "data": {
                "result_url": f"/static/ar_results/{username}/{result_filename}",
                "result_filename": result_filename,
                "overlay_used": overlay_name,
                "processing_time": datetime.now().isoformat(),
                "credits_remaining": new_credit_balance,
                "user_folder": username,
                "file_size": final_result_path.stat().st_size if final_result_path.exists() else 0
            }
        }
        
        logger.info(f"AR photo successful for {username}: {final_result_path}, credits remaining: {new_credit_balance}")
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"AR photo error for user {current_user.get('username', 'unknown')}: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan dalam pembuatan AR photo")
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")

# ===== TEMPLATE & AR ENDPOINTS =====

@app.get("/api/templates")
async def list_templates():
    try:
        templates = []
        if Config.TEMPLATE_DIR.exists():
            for file_path in Config.TEMPLATE_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in Config.ALLOWED_EXTENSIONS:
                    templates.append({
                        "name": file_path.name,
                        "path": f"/static/templates/{file_path.name}"
                    })
        
        # If no templates exist, create sample data
        if not templates:
            sample_templates = [
                {"name": "superhero.jpg", "path": "/static/templates/superhero.jpg"},
                {"name": "princess.jpg", "path": "/static/templates/princess.jpg"},
                {"name": "warrior.jpg", "path": "/static/templates/warrior.jpg"},
                {"name": "cartoon.jpg", "path": "/static/templates/cartoon.jpg"}
            ]
            templates = sample_templates
        
        return JSONResponse({
            "success": True,
            "templates": templates,
            "count": len(templates)
        })
    
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail="Gagal mengambil daftar template")

@app.post("/api/ar_upload")
async def ar_upload(webcam: UploadFile = File(...), template_name: str = Form(...)):
    try:
        clean_template_name = os.path.splitext(template_name)[0]
        filename = f"{uuid.uuid4().hex}_{clean_template_name}.png"
        save_path = Config.AR_RESULTS_DIR / filename

        with save_path.open("wb") as buffer:
            shutil.copyfileobj(webcam.file, buffer)

        logger.info(f"AR photo saved: {save_path}")
        return JSONResponse(content={"success": True, "filename": filename})

    except Exception as e:
        logger.error(f"AR upload error: {e}")
        return JSONResponse(content={"success": False, "error": str(e)})

@app.get("/api/ar/characters")
async def ar_characters_dynamic():
    """Dynamic AR characters from directory scan"""
    try:
        characters = []
        
        thumbnail_dir = Path("static/ar_assets/thumbnail")
        ar_assets_dir = Path("static/ar_assets")
        
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        ar_assets_dir.mkdir(parents=True, exist_ok=True)
        
        if thumbnail_dir.exists():
            for thumbnail_file in thumbnail_dir.iterdir():
                if thumbnail_file.is_file() and thumbnail_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    character_name = thumbnail_file.stem
                    
                    webm_file = ar_assets_dir / f"{character_name}.webm"
                    has_animation = webm_file.exists()
                    
                    character_data = {
                        "name": character_name,
                        "display_name": character_name.replace('_', ' ').title(),
                        "thumbnail": f"/static/ar_assets/thumbnail/{thumbnail_file.name}",
                        "has_animation": has_animation,
                        "animation_url": f"/static/ar_assets/{character_name}.webm" if has_animation else None,
                        "webm_url": f"/static/ar_assets/{character_name}.webm" if has_animation else None,
                        "type": "photo_ar"
                    }
                    
                    characters.append(character_data)
        
        if not characters:
            return {
                "success": True,
                "characters": [],
                "count": 0,
                "message": "No AR characters found. Add thumbnail + webm files."
            }
        
        return {
            "success": True,
            "characters": characters,
            "count": len(characters),
            "message": f"Found {len(characters)} AR characters"
        }
    
    except Exception as e:
        logger.error(f"Error scanning AR characters: {e}")
        return {
            "success": False,
            "error": str(e),
            "characters": [],
            "count": 0
        }

# ===== ADMIN ENDPOINTS =====

if ENHANCED_AUTH_AVAILABLE:
    
    @app.get("/api/admin/users")
    async def list_users(admin_user = Depends(admin_required)):
        """List all users with statistics (Admin only)"""
        try:
            with auth_service.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        u.id, u.username, u.role, u.credit_balance, u.created_at, 
                        u.last_login, u.is_active,
                        COUNT(DISTINCT p.id) as total_photos,
                        COUNT(DISTINCT CASE WHEN p.photo_type = 'face_swap' THEN p.id END) as face_swap_count,
                        COUNT(DISTINCT CASE WHEN p.photo_type = 'ar_photo' THEN p.id END) as ar_photo_count,
                        COALESCE(SUM(t.amount), 0) as total_spent,
                        MAX(p.created_at) as last_photo
                    FROM users u
                    LEFT JOIN photos p ON u.id = p.user_id
                    LEFT JOIN transactions t ON u.id = t.user_id AND t.status = 'settlement'
                    WHERE u.role != 'admin'
                    GROUP BY u.id
                    ORDER BY total_photos DESC
                """)
                
                users = []
                for row in cursor.fetchall():
                    users.append({
                        "id": row[0],
                        "username": row[1],
                        "role": row[2],
                        "credit_balance": row[3],
                        "created_at": row[4],
                        "last_login": row[5],
                        "is_active": bool(row[6]),
                        "total_photos": row[7],
                        "face_swap_count": row[8],
                        "ar_photo_count": row[9],
                        "total_spent": row[10],
                        "last_photo": row[11]
                    })
                
                return {"success": True, "users": users, "count": len(users)}
                
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve users")

    @app.get("/api/admin/stats/quick")
    async def get_quick_stats(admin_user = Depends(admin_required)):
        """Get quick statistics for admin dashboard"""
        try:
            with auth_service.db_manager.get_connection() as conn:
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
                total_users = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM photos WHERE photo_type = 'face_swap'")
                total_face_swap = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM photos WHERE photo_type = 'ar_photo'")
                total_ar_photos = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE status = 'settlement'")
                total_revenue = cursor.fetchone()[0]
                
                return {
                    "success": True,
                    "stats": {
                        "total_users": total_users,
                        "total_face_swap": total_face_swap,
                        "total_ar_photos": total_ar_photos,
                        "total_revenue": total_revenue
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting quick stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

    @app.post("/api/admin/users/{user_id}/add-credits")
    async def admin_add_credits(
        user_id: int, 
        credits: int = Form(...),
        reason: str = Form("Manual admin adjustment"),
        admin_user = Depends(admin_required)
    ):
        """Admin function to add credits to user"""
        
        if credits <= 0:
            raise HTTPException(status_code=400, detail="Credits must be positive")
        
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            username = user[0]
            
            cursor = conn.execute("""
                UPDATE users 
                SET credit_balance = credit_balance + ? 
                WHERE id = ?
            """, (credits, user_id))
            
            cursor = conn.execute("SELECT credit_balance FROM users WHERE id = ?", (user_id,))
            new_balance = cursor.fetchone()[0]
            
            conn.execute("""
                INSERT INTO transactions (user_id, order_id, amount, credits_added, status, settled_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, f"ADMIN-{uuid.uuid4().hex[:8]}", 0, credits, "admin_credit"))
            
            conn.commit()
        
        logger.info(f"Admin {admin_user['username']} added {credits} credits to {username}")
        
        return {
            "success": True,
            "message": f"Added {credits} credits to {username}",
            "new_balance": new_balance,
            "reason": reason
        }

# ===== UTILITY ENDPOINTS =====

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "AI Face Swap Studio is running"}

@app.get("/api/info")
async def app_info():
    """Get enhanced application information"""
    return {
        "app_name": "AI Face Swap Studio",
        "version": "2.1.0",
        "description": "Advanced Face Swapping API with Authentication and AR Photo",
        "enhanced_auth": ENHANCED_AUTH_AVAILABLE,
        "phase": "Phase 1 - Foundation" if ENHANCED_AUTH_AVAILABLE else "Basic",
        "features": [
            "Face Swapping",
            "AR Photo Overlay", 
            "Role-based Authentication" if ENHANCED_AUTH_AVAILABLE else "Basic Authentication",
            "Credit System Foundation" if ENHANCED_AUTH_AVAILABLE else "Session-based Limits",
            "QR Code Generation",
            "User Dashboard"
        ],
        "api_docs": "/docs"
    }

@app.get("/api/test/phase1")
async def test_phase1():
    """Test Phase 1 implementation"""
    try:
        test_results = []
        
        # Test database connection
        # try:
        #     with auth_service.db_manager.get_connection() as conn:
        #             "test": f"{'Enhanced' if ENHANCED_AUTH_AVAILABLE else 'Basic'} database connection",
        #             "status": "✅ PASS",
        #             "details": f"{user_count} users in database"
        #         })
        # except Exception as e:
        #     test_results.append({
        #         "test": "Database connection", 
        #         "status": "❌ FAIL",
        #         "details": str(e)
        #     })
        
        # Test enhanced auth service
        if ENHANCED_AUTH_AVAILABLE:
            try:
                schema_valid = validate_database_schema()
                test_results.append({
                    "test": "Enhanced auth service",
                    "status": "✅ PASS" if schema_valid else "❌ FAIL",
                    "details": "Schema validation " + ("passed" if schema_valid else "failed")
                })
            except Exception as e:
                test_results.append({
                    "test": "Enhanced auth service",
                    "status": "❌ FAIL", 
                    "details": str(e)
                })
        else:
            test_results.append({
                "test": "Enhanced auth service",
                "status": "⚠️ SKIP",
                "details": "Enhanced auth not available - run migration.py"
            })
        
        # Test models initialization
        global face_app, swapper
        if face_app and swapper:
            test_results.append({
                "test": "AI models loaded",
                "status": "✅ PASS",
                "details": "Face analysis and swapper models ready"
            })
        else:
            test_results.append({
                "test": "AI models loaded",
                "status": "❌ FAIL",
                "details": "Models not loaded properly"
            })
        
        # Test directory structure
        try:
            required_dirs = [Config.UPLOAD_DIR, Config.TEMPLATE_DIR, Config.RESULT_DIR]
            all_dirs_exist = all(d.exists() for d in required_dirs)
            test_results.append({
                "test": "Directory structure",
                "status": "✅ PASS" if all_dirs_exist else "❌ FAIL",
                "details": f"Required directories {'exist' if all_dirs_exist else 'missing'}"
            })
        except Exception as e:
            test_results.append({
                "test": "Directory structure",
                "status": "❌ FAIL",
                "details": str(e)
            })
        
        # Overall status
        all_passed = all("✅ PASS" in result["status"] for result in test_results)
        
        return {
            "phase": "Phase 1 - Foundation",
            "status": "✅ READY" if all_passed else "⚠️ PARTIAL" if ENHANCED_AUTH_AVAILABLE else "❌ NEEDS_MIGRATION",
            "enhanced_auth_available": ENHANCED_AUTH_AVAILABLE,
            "test_results": test_results,
            "next_phase": "Phase 2 - Credit System & Enhanced Photo Generation",
            "recommendations": [
                "Run migration.py to enable enhanced features" if not ENHANCED_AUTH_AVAILABLE else "Ready for Phase 2",
                "Test admin and user login flows",
                "Verify role-based redirects work",
                "Check existing functionality still works"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "phase": "Phase 1 - Foundation",
            "status": "❌ ERROR", 
            "error": str(e),
            "enhanced_auth_available": ENHANCED_AUTH_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/test/auth")
async def test_auth_system():
    """Test authentication system"""
    try:
        test_results = []
        
        if ENHANCED_AUTH_AVAILABLE:
            # Test enhanced auth features
            with auth_service.db_manager.get_connection() as conn:
                # Check admin users
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
                admin_count = cursor.fetchone()[0]
                test_results.append({
                    "test": "Admin users exist",
                    "status": "✅ PASS" if admin_count > 0 else "❌ FAIL",
                    "details": f"{admin_count} admin users found"
                })
                
                # Check regular users
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
                user_count = cursor.fetchone()[0]
                test_results.append({
                    "test": "Regular users exist",
                    "status": "✅ PASS" if user_count > 0 else "❌ FAIL",
                    "details": f"{user_count} regular users found"
                })
                
                # Check transactions table
                cursor = conn.execute("SELECT COUNT(*) FROM transactions")
                transactions_count = cursor.fetchone()[0]
                test_results.append({
                    "test": "Transactions table",
                    "status": "✅ PASS",
                    "details": f"{transactions_count} transactions found"
                })
        else:
            test_results.append({
                "test": "Enhanced authentication",
                "status": "❌ NOT_AVAILABLE",
                "details": "Run migration.py to enable enhanced auth"
            })
        
        return {
            "success": True,
            "auth_system": "Enhanced" if ENHANCED_AUTH_AVAILABLE else "Basic",
            "test_results": test_results,
            "sample_accounts": [
                {"username": "admin", "password": "admin123", "role": "admin"},
                {"username": "cbt", "password": "cbt123", "role": "user"},
                {"username": "bsd", "password": "bsd123", "role": "user"}
            ] if ENHANCED_AUTH_AVAILABLE else [
                {"username": "demo", "password": "demo123", "role": "user"}
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/count-files")
def count_files():
    """Count files in result directories"""
    try:
        face_swap_path = "static/results"
        ar_photo_path = "static/ar_results"
        
        face_files = []
        ar_files = []
        
        if os.path.exists(face_swap_path):
            for root, dirs, files in os.walk(face_swap_path):
                face_files.extend([f for f in files if os.path.isfile(os.path.join(root, f))])
        
        if os.path.exists(ar_photo_path):
            for root, dirs, files in os.walk(ar_photo_path):
                ar_files.extend([f for f in files if os.path.isfile(os.path.join(root, f))])

        return {
            "success": True,
            "faceSwapCount": len(face_files),
            "arPhotoCount": len(ar_files),
            "totalFiles": len(face_files) + len(ar_files)
        }
    except Exception as e:
        logger.error(f"Error counting files: {e}")
        return {
            "success": False,
            "faceSwapCount": 0,
            "arPhotoCount": 0,
            "totalFiles": 0,
            "error": str(e)
        }

# ===== USER HISTORY ENDPOINTS =====

@app.get("/api/history")
async def get_user_history(current_user = Depends(get_current_user)):
    """Get user photo history"""
    try:
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT filename, photo_type, template_name, created_at, file_path
                FROM photos 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
            """, (current_user["id"],))
            
            history = cursor.fetchall()
            
            history_list = []
            for record in history:
                filename, photo_type, template_name, created_at, file_path = record
                
                # Determine result URL based on photo type and user folder
                if photo_type == "ar_photo":
                    result_url = f"/static/ar_results/{current_user['username']}/{filename}"
                else:
                    result_url = f"/static/results/{current_user['username']}/{filename}"
                
                history_list.append({
                    "filename": filename,
                    "photo_type": photo_type,
                    "template_name": template_name,
                    "result_url": result_url,
                    "created_at": created_at
                })
            
            return {
                "success": True,
                "history": history_list,
                "count": len(history_list)
            }
    
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        raise HTTPException(status_code=500, detail="Gagal mengambil riwayat")

@app.delete("/api/results/{filename}")
async def delete_result(filename: str, current_user = Depends(get_current_user)):
    """Delete user's result file"""
    try:
        # Check if file belongs to user
        with auth_service.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT photo_type, file_path FROM photos 
                WHERE user_id = ? AND filename = ?
            """, (current_user["id"], filename))
            
            photo = cursor.fetchone()
            
            if not photo:
                raise HTTPException(status_code=404, detail="File tidak ditemukan atau bukan milik Anda")
            
            photo_type, file_path = photo
            
            # Delete from filesystem
            result_path = Path(file_path)
            if result_path.exists():
                result_path.unlink()
                logger.info(f"Deleted result file: {result_path}")
            
            # Delete from database
            conn.execute("DELETE FROM photos WHERE user_id = ? AND filename = ?", (current_user["id"], filename))
            conn.commit()
        
        return JSONResponse({
            "success": True,
            "message": f"File '{filename}' berhasil dihapus"
        })
    
    except Exception as e:
        logger.error(f"Error deleting result file: {e}")
        raise HTTPException(status_code=500, detail="Gagal menghapus file")

# ===== STARTUP MESSAGE =====

@app.on_event("startup")
async def startup_message():
    """Display startup information"""
    logger.info("=" * 60)
    logger.info("🚀 AI FACE SWAP STUDIO - STARTUP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"📡 Server: http://localhost:5000")
    logger.info(f"📚 API Docs: http://localhost:5000/docs")
    logger.info(f"🔐 Auth System: {'Enhanced' if ENHANCED_AUTH_AVAILABLE else 'Basic'}")
    
    if ENHANCED_AUTH_AVAILABLE:
        logger.info("✅ Enhanced Features Available:")
        logger.info("   - Role-based authentication (admin/user)")
        logger.info("   - Credit system foundation")
        logger.info("   - User-specific file organization")
        logger.info("   - Admin endpoints preview")
        logger.info("🔗 Test URLs:")
        logger.info("   - Login: http://localhost:5000/login")
        logger.info("   - Admin Dashboard: http://localhost:5000/dashboard_admin")
        logger.info("   - User Dashboard: http://localhost:5000/dashboard")
        logger.info("   - Phase 1 Test: http://localhost:5000/api/test/phase1")
    else:
        logger.info("⚠️  Enhanced Features Not Available")
        logger.info("   Run migration.py to enable:")
        logger.info("   - Role-based authentication")
        logger.info("   - Credit system")
        logger.info("   - Admin dashboard")
        logger.info("   - User management")
    
    logger.info("=" * 60)

# ===== MAIN APPLICATION ENTRY POINT =====

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AI Face Swap Studio...")
    
    uvicorn.run(
        "main:app",
        host="localhost",
        port=5000,
        reload=True,
        log_level="info"
    )