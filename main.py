from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Depends, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import uuid
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
import mimetypes
import requests
import shutil
import traceback

# Import authentication modules
import sqlite3
import hashlib
import secrets
import jwt
from pydantic import BaseModel

# Import payment integration
from midtrans_config import core_api

# =============================================
# CONFIGURATION & MODELS
# =============================================

class Config:
    UPLOAD_DIR = Path("static/uploads")
    TEMPLATE_DIR = Path("static/templates")
    RESULT_DIR = Path("static/results")
    FRAME_DIR = Path("static/images")
    PAGES_DIR = Path("pages")
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

# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# =============================================
# LOGGING SETUP
# =============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================
# GLOBAL VARIABLES
# =============================================

face_app = None
swapper = None
ESP32_IP = "10.65.124.246"
lampu_status = "off"

# =============================================
# DATABASE MANAGEMENT
# =============================================

DB_PATH = Path("face_swap.db")

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Users table with enhanced schema
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
            
            # Transactions table for payment tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    order_id TEXT UNIQUE NOT NULL,
                    amount INTEGER NOT NULL,
                    credits_added INTEGER DEFAULT 3,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Photos table (enhanced face_swap_history)
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
            
            # Keep legacy table for backward compatibility
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_swap_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    template_name TEXT NOT NULL,
                    result_filename TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Settings table for dynamic configuration
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_photos ON photos(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON transactions(order_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photo_type ON photos(photo_type)")
            
            # Insert default settings if not exist
            default_settings = [
                ('price_per_3_photos', '8', 'Harga untuk 3 foto'),
                ('credits_per_payment', '3', 'Credit per pembayaran'),
                ('photos_per_session', '3', 'Foto per session'),
                ('admin_username', 'admin', 'Default admin username'),
                ('admin_password', 'admin123', 'Default admin password')
            ]
            
            for key, value, desc in default_settings:
                conn.execute(
                    "INSERT OR IGNORE INTO settings (key_name, value, description) VALUES (?, ?, ?)",
                    (key, value, desc)
                )
            
            # Create default admin user if not exists
            cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            if cursor.fetchone()[0] == 0:
                # Create admin user
                admin_password = "admin123"
                salt = secrets.token_hex(32)
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    admin_password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                ).hex()
                
                conn.execute("""
                    INSERT INTO users (username, password_hash, salt, role, credit_balance)
                    VALUES (?, ?, ?, ?, ?)
                """, ("admin", password_hash, salt, "admin", 999999))
            
            conn.commit()
            logger.info("Database initialized successfully")

    def get_connection(self):
        return sqlite3.connect(self.db_path)

# =============================================
# AUTHENTICATION SERVICE
# =============================================

class AuthService:
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
    
    def create_jwt_token(self, user_id: int, username: str, role: str) -> str:
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
    
    def register_user(self, user_data: UserCreate) -> dict:
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (user_data.username,)
            )
            
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username sudah digunakan"
                )
            
            if len(user_data.password) < 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password minimal 4 karakter"
                )
            
            password_hash, salt = self.hash_password(user_data.password)
            
            cursor = conn.execute("""
                INSERT INTO users (username, password_hash, salt, role, credit_balance)
                VALUES (?, ?, ?, ?, ?)
            """, (user_data.username, password_hash, salt, "user", 0))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                "success": True,
                "message": "User berhasil didaftarkan",
                "user_id": user_id
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
            
            token = self.create_jwt_token(user_id, username, role)
            
            # Update last login
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_id,))
            conn.commit()
            
            # Return role-based redirect
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
                "role": user[2],
                "credit_balance": user[3],
                "created_at": user[4],
                "last_login": user[5]
            }

# =============================================
# APPLICATION LIFESPAN & INITIALIZATION
# =============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global face_app, swapper
    
    try:
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
        
        logger.info("Application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Application shutdown")

# =============================================
# FASTAPI APPLICATION SETUP
# =============================================

app = FastAPI(
    title="AI Face Swap Studio",
    description="Multi-Role Photobooth System with Admin Dashboard",
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

# Initialize services
auth_service = AuthService()
security = HTTPBearer(auto_error=False)

# =============================================
# AUTHENTICATION DEPENDENCIES
# =============================================

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    token = credentials.credentials
    return auth_service.get_user_by_token(token)

async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        return auth_service.get_user_by_token(token)
    except HTTPException:
        return None

async def admin_required(current_user = Depends(get_current_user)):
    """Require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def user_required(current_user = Depends(get_current_user)):
    """Require user or admin role"""
    if current_user.get("role") not in ["admin", "user"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User access required"
        )
    return current_user

async def check_user_credits(current_user = Depends(get_current_user)):
    """Check if user has sufficient credits"""
    if current_user["credit_balance"] < 1:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits. Please make a payment."
        )
    return current_user

# =============================================
# UTILITY FUNCTIONS
# =============================================

class FaceSwapError(Exception):
    pass

class ValidationError(Exception):
    pass

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

def generate_unique_filename(original_filename: str, username: str, prefix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_ext = Path(original_filename).suffix.lower()
    
    safe_filename = f"{username}_{timestamp}_{unique_id}{file_ext}"
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
        </body></html>
        """)

# Face processing functions (simplified for brevity)
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
        
        if img_src is None or img_dst is None:
            raise FaceSwapError("Tidak dapat membaca gambar")
        
        faces_src = detect_faces(src_path)
        faces_dst = detect_faces(dst_path)
        
        if len(faces_src) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada gambar sumber")
        if len(faces_dst) == 0:
            raise FaceSwapError("Tidak ada wajah yang terdeteksi pada template")
        
        result = swapper.get(img_dst.copy(), faces_dst[0], faces_src[0], paste_back=True)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), result)
        
        if not success:
            raise FaceSwapError("Gagal menyimpan hasil face swap")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Face swap error: {e}")
        raise FaceSwapError(f"Error dalam proses face swap: {e}")

# =============================================
# FRONTEND ROUTES
# =============================================

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

@app.get("/result", response_class=HTMLResponse)
async def result_page():
    return serve_html_page("result")

@app.get("/manipulasi", response_class=HTMLResponse)
async def manipulasi_page():
    return serve_html_page("manipulasi")

@app.get("/payment", response_class=HTMLResponse)
async def payment_page():
    return serve_html_page("payment")

@app.get("/ar_manipulasi", response_class=HTMLResponse)
async def ar_manipulasi_page():
    return serve_html_page("ar_manipulasi")

@app.get("/ar-character", response_class=HTMLResponse)
async def ar_character_page():
    return serve_html_page("ar_character")

@app.get("/ar_camera", response_class=HTMLResponse)
async def ar_camera_page():
    return serve_html_page("ar_camera")

@app.get("/ar_payment", response_class=HTMLResponse)
async def ar_payment_page():
    return serve_html_page("ar_payment")

@app.get("/ar-result", response_class=HTMLResponse)
async def ar_result_page():
    return serve_html_page("ar_result")

# =============================================
# AUTHENTICATION API ROUTES
# =============================================

@app.post("/api/register")
async def register(user_data: UserCreate):
    return auth_service.register_user(user_data)

@app.post("/api/login")
async def login(login_data: UserLogin):
    return auth_service.login_user(login_data)

@app.post("/api/logout")
async def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    return {"success": True, "message": "Logout berhasil"}

@app.get("/api/me")
async def get_me(current_user = Depends(get_current_user)):
    return {"success": True, "user": current_user}

# =============================================
# PAYMENT API ROUTES
# =============================================

@app.get("/api/qris/token")
async def generate_qris_token(current_user = Depends(get_current_user_optional)):
    """Generate QRIS token - now works without authentication for easier testing"""
    try:
        order_id = f"ORDER-{uuid.uuid4().hex[:12]}"
        
        # Record transaction with user_id if authenticated
        user_id = current_user["id"] if current_user else None
        
        with auth_service.db_manager.get_connection() as conn:
            conn.execute("""
                INSERT INTO transactions (user_id, order_id, amount, credits_added, status)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, order_id, 8, 3, "pending"))
            conn.commit()
        
        payload = {
            "payment_type": "qris",
            "transaction_details": {
                "order_id": order_id,
                "gross_amount": 8,
            },
            "qris": {
                "acquirer": "gopay"
            }
        }

        result = core_api.charge(payload)
        logger.info(f"Midtrans charge successful for order_id: {order_id}")

        # Get QR URL from actions
        actions = result.get("actions", [])
        qris_url = next((a["url"] for a in actions if a.get("name") == "generate-qr-code"), None)

        if not qris_url:
            logger.error(f"QRIS URL not found in Midtrans response for order_id: {order_id}")
            return JSONResponse(
                status_code=400, 
                content={"success": False, "error": "QRIS URL tidak ditemukan dalam respons Midtrans"}
            )

        return {"success": True, "qris_url": qris_url, "order_id": order_id}

    except requests.exceptions.RequestException as e:
        logger.error(f"Midtrans RequestException for order_id {order_id}: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": "Gagal menghubungi API Midtrans"}
        )

    except Exception as e:
        logger.error(f"General Exception during Midtrans charge: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": "Terjadi kesalahan internal saat memproses pembayaran"}
        )

@app.get("/api/qris/status")
async def check_qris_status(order_id: str):
    """Check QRIS payment status and auto-add credits"""
    try:
        status = core_api.transactions.status(order_id)
        
        if status.get("transaction_status") == "settlement":
            # Add credits to user
            with auth_service.db_manager.get_connection() as conn:
                # Find transaction and user
                cursor = conn.execute("SELECT user_id FROM transactions WHERE order_id = ?", (order_id,))
                transaction = cursor.fetchone()
                
                if transaction and transaction[0]:  # If user_id exists
                    user_id = transaction[0]
                    credits_to_add = 3
                    
                    # Add credits
                    conn.execute(
                        "UPDATE users SET credit_balance = credit_balance + ? WHERE id = ?", 
                        (credits_to_add, user_id)
                    )
                    
                    # Update transaction status
                    conn.execute(
                        "UPDATE transactions SET status = 'settlement', settled_at = CURRENT_TIMESTAMP WHERE order_id = ?",
                        (order_id,)
                    )
                    conn.commit()
                    logger.info(f"Added {credits_to_add} credits to user {user_id} for order {order_id}")
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking QRIS status: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============================================
# TEMPLATES & PHOTO GENERATION API ROUTES
# =============================================

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
        
        # Fallback demo templates
        if not templates:
            templates = [
                {"name": "superhero.jpg", "path": "/static/templates/superhero.jpg"},
                {"name": "princess.jpg", "path": "/static/templates/princess.jpg"},
                {"name": "warrior.jpg", "path": "/static/templates/warrior.jpg"},
                {"name": "cartoon.jpg", "path": "/static/templates/cartoon.jpg"}
            ]
        
        return JSONResponse({
            "success": True,
            "templates": templates,
            "count": len(templates)
        })
    
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengambil daftar template"
        )

@app.post("/api/swap")
async def swap_faces_api(
    template_name: str = Form(...),
    webcam: UploadFile = File(...),
    source: Optional[UploadFile] = File(None),
    apply_frame: bool = Form(True),
    current_user = Depends(check_user_credits)
):
    """Enhanced face swap with credit system and user-specific folders"""
    temp_files = []
    
    try:
        validate_file(webcam)
        if source:
            validate_file(source)
        
        template_path = Config.TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template '{template_name}' tidak ditemukan"
            )
        
        username = current_user["username"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Save webcam file
        webcam_filename = generate_unique_filename(webcam.filename, username, "webcam_")
        webcam_path = Config.UPLOAD_DIR / webcam_filename
        await save_uploaded_file(webcam, webcam_path)
        temp_files.append(webcam_path)
        
        # Handle source file
        source_path = webcam_path
        if source:
            source_filename = generate_unique_filename(source.filename, username, "source_")
            source_path = Config.UPLOAD_DIR / source_filename
            await save_uploaded_file(source, source_path)
            temp_files.append(source_path)
        
        # Create user-specific result directory
        user_result_dir = Config.RESULT_DIR / username
        user_result_dir.mkdir(parents=True, exist_ok=True)
        
        result_filename = f"{username}_{timestamp}_{unique_id}.png"
        result_path = user_result_dir / result_filename
        
        # Process face swap
        logger.info(f"Starting face swap: {source_path} -> {template_path}")
        swap_result_path = swap_faces(source_path, template_path, result_path)
        
        # Apply frame if requested
        if apply_frame:
            frame_path = Config.FRAME_DIR / "frame1.png"
            if frame_path.exists():
                # Apply frame overlay (simplified implementation)
                pass
        
        # Deduct credit and record photo
        with auth_service.db_manager.get_connection() as conn:
            # Deduct credit
            conn.execute(
                "UPDATE users SET credit_balance = credit_balance - 1 WHERE id = ?",
                (current_user["id"],)
            )
            
            # Record photo in new photos table
            conn.execute("""
                INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, credits_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (current_user["id"], result_filename, "face_swap", template_name, str(result_path), 1))
            
            # Also record in legacy table for backward compatibility
            conn.execute("""
                INSERT INTO face_swap_history (user_id, template_name, result_filename)
                VALUES (?, ?, ?)
            """, (current_user["id"], template_name, result_filename))
            
            conn.commit()
        
        response_data = {
            "success": True,
            "message": "Face swap berhasil dilakukan",
            "data": {
                "result_url": f"/static/results/{username}/{result_filename}",
                "result_filename": result_filename,
                "template_used": template_name,
                "faces_detected": {
                    "source": len(detect_faces(source_path)),
                    "template": len(detect_faces(template_path))
                },
                "frame_applied": apply_frame,
                "processing_time": datetime.now().isoformat(),
                "credits_remaining": current_user["credit_balance"] - 1
            }
        }
        
        logger.info(f"Face swap successful: {swap_result_path}")
        return JSONResponse(response_data)
    
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except FaceSwapError as e:
        logger.error(f"Face swap error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in face swap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan internal pada server"
        )
    
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")

# =============================================
# AR PHOTO API ROUTES
# =============================================

@app.get("/api/ar/characters")
async def ar_characters_dynamic():
    """Dynamic AR characters from directory scan"""
    try:
        characters = []
        
        # Define directories  
        thumbnail_dir = Path("static/ar_assets/thumbnail")
        ar_assets_dir = Path("static/ar_assets")
        
        # Create directories if they don't exist
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        ar_assets_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan thumbnail directory for real images
        if thumbnail_dir.exists():
            for thumbnail_file in thumbnail_dir.iterdir():
                if thumbnail_file.is_file() and thumbnail_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    character_name = thumbnail_file.stem
                    
                    # Look for corresponding webm animation file
                    webm_file = ar_assets_dir / f"{character_name}.webm"
                    has_animation = webm_file.exists()
                    
                    character_data = {
                        "name": character_name,
                        "display_name": character_name.replace('_', ' ').title(),
                        "thumbnail": f"/static/ar_assets/thumbnail/{thumbnail_file.name}",
                        "has_animation": has_animation,
                        "animation_url": f"/static/ar_assets/{character_name}.webm" if has_animation else None,
                        "type": "photo_ar"
                    }
                    
                    characters.append(character_data)
                    logger.info(f"AR Character found: {character_name} (webm: {has_animation})")
        
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

@app.post("/api/ar_upload")
async def ar_upload(
    webcam: UploadFile = File(...), 
    template_name: str = Form(...),
    current_user = Depends(get_current_user_optional)
):
    """Upload AR photo with optional user tracking"""
    try:
        # Clean template name
        clean_template_name = os.path.splitext(template_name)[0]

        # Generate unique filename
        if current_user:
            username = current_user["username"]
            filename = f"{username}_{uuid.uuid4().hex}_{clean_template_name}.png"
            
            # Create user-specific directory
            user_result_dir = Config.AR_RESULTS_DIR / username
            user_result_dir.mkdir(parents=True, exist_ok=True)
            save_path = user_result_dir / filename
        else:
            filename = f"{uuid.uuid4().hex}_{clean_template_name}.png"
            save_path = Config.AR_RESULTS_DIR / filename

        # Save file
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(webcam.file, buffer)

        # Record in database if user is authenticated
        if current_user:
            with auth_service.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, credits_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (current_user["id"], filename, "ar_photo", template_name, str(save_path), 0))
                conn.commit()

        logger.info(f"AR photo saved: {save_path}")

        return JSONResponse(content={"success": True, "filename": filename})

    except Exception as e:
        logger.error(f"Error during AR upload: {e}")
        return JSONResponse(content={"success": False, "error": str(e)})

# =============================================
# USER MANAGEMENT & ADMIN API ROUTES
# =============================================

@app.get("/api/user/credits")
async def get_user_credits(current_user = Depends(get_current_user)):
    """Get current user's credit balance"""
    return {
        "success": True,
        "credits": current_user["credit_balance"],
        "username": current_user["username"]
    }

@app.get("/api/history")
async def get_user_history(current_user = Depends(get_current_user)):
    """Get user's photo generation history"""
    try:
        with auth_service.db_manager.get_connection() as conn:
            # Get from new photos table
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
                
                # Construct result URL based on photo type and user folder
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengambil riwayat"
        )

@app.get("/api/count-files")
async def count_files():
    """Count total files for dashboard statistics"""
    try:
        face_swap_count = 0
        ar_photo_count = 0
        
        # Count files in results directory
        if Config.RESULT_DIR.exists():
            for user_dir in Config.RESULT_DIR.iterdir():
                if user_dir.is_dir():
                    face_swap_count += len([f for f in user_dir.iterdir() if f.is_file()])
        
        # Count files in ar_results directory  
        if Config.AR_RESULTS_DIR.exists():
            for user_dir in Config.AR_RESULTS_DIR.iterdir():
                if user_dir.is_dir():
                    ar_photo_count += len([f for f in user_dir.iterdir() if f.is_file()])
        
        return {
            "faceSwapCount": face_swap_count,
            "arPhotoCount": ar_photo_count
        }
    
    except Exception as e:
        logger.error(f"Error counting files: {e}")
        return {"faceSwapCount": 0, "arPhotoCount": 0}

# =============================================
# ADMIN DASHBOARD API ROUTES
# =============================================

@app.get("/api/admin/dashboard/stats")
async def get_dashboard_stats(admin_user = Depends(admin_required)):
    """Get dashboard statistics for admin"""
    try:
        with auth_service.db_manager.get_connection() as conn:
            # Basic counts
            total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0]
            total_face_swap = conn.execute("SELECT COUNT(*) FROM photos WHERE photo_type = 'face_swap'").fetchone()[0]
            total_ar_photos = conn.execute("SELECT COUNT(*) FROM photos WHERE photo_type = 'ar_photo'").fetchone()[0]
            total_revenue = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE status = 'settlement'").fetchone()[0]
            
            # Today's stats
            today_face_swap = conn.execute("SELECT COUNT(*) FROM photos WHERE photo_type = 'face_swap' AND DATE(created_at) = DATE('now')").fetchone()[0]
            today_ar_photos = conn.execute("SELECT COUNT(*) FROM photos WHERE photo_type = 'ar_photo' AND DATE(created_at) = DATE('now')").fetchone()[0]
            today_revenue = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE status = 'settlement' AND DATE(settled_at) = DATE('now')").fetchone()[0]
            today_transactions = conn.execute("SELECT COUNT(*) FROM transactions WHERE status = 'settlement' AND DATE(settled_at) = DATE('now')").fetchone()[0]
            
            return {
                "success": True,
                "total_users": total_users,
                "total_face_swap": total_face_swap,
                "total_ar_photos": total_ar_photos,
                "total_revenue": total_revenue,
                "today_stats": {
                    "face_swap": today_face_swap,
                    "ar_photos": today_ar_photos,
                    "revenue": today_revenue,
                    "transactions": today_transactions
                }
            }
    
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengambil statistik dashboard"
        )

@app.get("/api/admin/users")
async def list_users(admin_user = Depends(admin_required)):
    """List all users with statistics for admin"""
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
                WHERE u.role = 'user'
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
            
            return {"success": True, "users": users}
    
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal mengambil daftar user"
        )

@app.post("/api/admin/users/{user_id}/reset-credits")
async def reset_user_credits(
    user_id: int, 
    credits: int, 
    admin_user = Depends(admin_required)
):
    """Reset user credits (admin only)"""
    try:
        with auth_service.db_manager.get_connection() as conn:
            conn.execute(
                "UPDATE users SET credit_balance = ? WHERE id = ? AND role = 'user'",
                (credits, user_id)
            )
            conn.commit()
        
        return {"success": True, "message": f"Credits reset to {credits}"}
    
    except Exception as e:
        logger.error(f"Error resetting credits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal reset credits"
        )

# =============================================
# HARDWARE CONTROL API ROUTES
# =============================================

@app.get("/relay")
async def relay_control(state: str):
    """Control ESP32 relay for lighting"""
    if state not in ["on", "off"]:
        return JSONResponse({"success": False, "message": "Invalid state"}, status_code=400)
    
    try:
        resp = requests.get(f"http://{ESP32_IP}/relay?state={state}", timeout=2)
        if resp.status_code == 200:
            global lampu_status
            lampu_status = state
            return {"success": True, "status": state}
        else:
            return JSONResponse({"success": False, "message": "ESP error"}, status_code=500)
    except:
        return JSONResponse({"success": False, "message": "ESP unreachable"}, status_code=500)

@app.get("/status")
async def get_status():
    """Get current lamp status"""
    return {"status": lampu_status}

@app.get("/reset-lampu")
async def reset_lampu():
    """Reset lamp status"""
    global lampu_status
    lampu_status = "off"
    return {"success": True}

# =============================================
# UTILITY & INFO API ROUTES
# =============================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "AI Face Swap Studio is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/info")
async def app_info():
    """Get application information"""
    return {
        "app_name": "AI Face Swap Studio",
        "version": "2.1.0",
        "description": "Multi-Role Photobooth System with Admin Dashboard",
        "features": [
            "Role-based Authentication",
            "Credit System",
            "Face Swapping",
            "AR Photo Overlay",
            "Admin Dashboard",
            "User Management",
            "Payment Integration"
        ],
        "api_docs": "/docs"
    }

# =============================================
# APPLICATION ENTRY POINT
# =============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )