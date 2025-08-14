# ar_photo.py - FIXED VERSION
"""
AR Photo functionality module with fixes for overlay and display issues
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import logging
import uuid
import qrcode
from io import BytesIO
import base64
import mimetypes
import aiofiles
import shutil
import traceback


# Setup logging
logger = logging.getLogger(__name__)

# AR Photo specific configuration
class ARConfig:
    AR_ASSETS_DIR = Path("static/ar_assets")
    COUNTDOWN_DIR = Path("static/ar_assets/countdown")
    THUMBNAIL_DIR = Path("static/ar_assets/thumbnail")
    AR_RESULTS_DIR = Path("static/ar_results")
    UPLOAD_DIR = Path("static/uploads")
    FRAME_DIR = Path("static/ar_assets/frame_full")  # Corrected path for character frames

    
    # AR specific settings
    WEBM_DURATION = 3  # seconds for webm character display
    AR_COUNTDOWN_DURATION = 5  # seconds for countdown
    RESULT_WAIT_DURATION = 5  # seconds before redirect to result
    QR_CODE_SIZE = 10
    QR_BORDER = 4
    
    # File settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/bmp'}
    
    # Domain for QR codes (update this for production)
    DOMAIN_URL = "https://https://faceswap.platinumphotobooth.my.id/"

# Create router for AR Photo endpoints
router = APIRouter(prefix="/api/ar", tags=["AR Photo"])

# Authentication setup
security = HTTPBearer(auto_error=False)

class ARPhotoError(Exception):
    """Custom exception for AR Photo operations"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise ValidationError("Filename tidak boleh kosong")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ARConfig.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Ekstensi file tidak didukung. "
            f"Maksimal {ARConfig.MAX_FILE_SIZE // (1024*1024)}MB"
        )

def generate_unique_filename(original_name: str, prefix: str = "") -> str:
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_ext = Path(original_name).suffix.lower()
    return f"{prefix}{timestamp}_{unique_id}{file_ext}"

async def save_uploaded_file(upload_file: UploadFile, save_path: Path) -> Path:
    """Save uploaded file to specified path"""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await upload_file.read()
        
        if len(content) > ARConfig.MAX_FILE_SIZE:
            raise ValidationError(
                f"File terlalu besar. "
                f"Maksimal {ARConfig.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        async with aiofiles.open(save_path, 'wb') as f:
            await f.write(content)
        
        logger.info(f"File saved: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error saving file {save_path}: {e}")
        if save_path.exists():
            save_path.unlink()
        raise

def generate_qr_code(data: str, size: int = ARConfig.QR_CODE_SIZE) -> str:
    """Generate QR code and return as base64 string"""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=size,
            border=ARConfig.QR_BORDER,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffered = BytesIO()
        qr_img.save(buffered, format="PNG")
        qr_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{qr_base64}"
    
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        return ""

def apply_ar_overlay(base_image_path: Path, overlay_path: Path, output_path: Path) -> Path:
    """Apply AR overlay to captured photo - FIXED VERSION"""
    try:
        logger.info(f"Applying AR overlay: base={base_image_path}, overlay={overlay_path}, output={output_path}")

        # Check if overlay file exists
        if not overlay_path.exists():
            logger.warning(f"Overlay file not found: {overlay_path}")
            # FIXED: Ensure output directory exists before copying
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_image_path, output_path)
            return output_path

        # Read base image
        base_img = cv2.imread(str(base_image_path), cv2.IMREAD_UNCHANGED)
        if base_img is None:
            logger.error(f"Cannot read base image: {base_image_path}")
            raise ARPhotoError(f"Cannot read base image: {base_image_path}")

        logger.info(f"Base image loaded: shape={base_img.shape}")

        # Read overlay image with alpha channel support
        overlay_img = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            logger.warning(f"Cannot read overlay image: {overlay_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_image_path, output_path)
            return output_path

        logger.info(f"Overlay image loaded: shape={overlay_img.shape}")

        # FIXED: Ensure base image has 3 channels (RGB)
        if len(base_img.shape) == 3 and base_img.shape[2] == 4:
            # Convert RGBA to RGB
            base_img = cv2.cvtColor(base_img, cv2.COLOR_BGRA2BGR)
        elif len(base_img.shape) == 2:
            # Convert grayscale to RGB
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

        # Resize overlay to match base image dimensions
        h, w = base_img.shape[:2]
        overlay_img = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)
        logger.info(f"Overlay resized to: {overlay_img.shape}")

        # FIXED: Apply overlay with proper alpha blending
        if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 4:
            # Overlay has alpha channel
            logger.info("Applying overlay with alpha channel")
            
            # Extract alpha channel (0-255)
            alpha = overlay_img[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            
            # Apply alpha blending for each channel
            for c in range(3):  # RGB channels
                base_img[:, :, c] = (alpha * overlay_img[:, :, c] + 
                                   alpha_inv * base_img[:, :, c])
            
            result = base_img.astype(np.uint8)
        else:
            # Simple overlay without alpha channel
            logger.info("Applying simple overlay blend")
            if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 3:
                result = cv2.addWeighted(base_img, 0.7, overlay_img, 0.3, 0)
            else:
                # If overlay is grayscale, convert to BGR
                overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(base_img, 0.7, overlay_bgr, 0.3, 0)

        # FIXED: Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        success = cv2.imwrite(str(output_path), result)
        
        if not success:
            logger.error(f"Failed to save result to: {output_path}")
            raise ARPhotoError("Failed to save AR overlay result")

        logger.info(f"AR overlay applied successfully: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"AR overlay error: {e}")
        # FIXED: Ensure fallback works properly
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_image_path, output_path)
            logger.info(f"Fallback: copied original image to {output_path}")
        except Exception as fallback_error:
            logger.error(f"Fallback copy failed: {fallback_error}")
            raise ARPhotoError(f"Complete failure in AR overlay: {e}")
        
        return output_path

def create_ar_directories():
    """Create necessary directories for AR Photo functionality"""
    directories = [
        ARConfig.AR_ASSETS_DIR,
        ARConfig.COUNTDOWN_DIR,
        ARConfig.THUMBNAIL_DIR,
        ARConfig.AR_RESULTS_DIR,
        ARConfig.UPLOAD_DIR,
        ARConfig.FRAME_DIR  # FIXED: Ensure frame directory is created
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"AR directory created/verified: {directory}")

# Authentication dependency
async def get_current_user_ar(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Authentication for AR endpoints"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        from main import auth_service
        token = credentials.credentials
        return auth_service.get_user_by_token(token)
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service not available"
        )

# ===== AR PHOTO ROUTES =====

@router.get("/characters")
async def list_ar_characters():
    """FIXED: List available AR character thumbnails and webm files"""
    try:
        characters = []
        
        # Ensure directories exist
        ARConfig.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        ARConfig.AR_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Scanning thumbnails in: {ARConfig.THUMBNAIL_DIR}")
        
        if ARConfig.THUMBNAIL_DIR.exists():
            for thumbnail_file in ARConfig.THUMBNAIL_DIR.iterdir():
                if thumbnail_file.is_file() and thumbnail_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    character_name = thumbnail_file.stem  # filename without extension
                    
                    # Look for corresponding webm file
                    webm_file = ARConfig.AR_ASSETS_DIR / f"{character_name}.webm"
                    
                    # FIXED: Also look for overlay frame
                    frame_file = ARConfig.FRAME_DIR / f"{character_name}_frame.png"
                    
                    character_data = {
                        "name": character_name,
                        "display_name": character_name.replace('_', ' ').title(),
                        "thumbnail": f"/static/ar_assets/thumbnail/{thumbnail_file.name}",
                        "has_animation": webm_file.exists(),
                        "animation_url": f"/static/ar_assets/{character_name}.webm" if webm_file.exists() else None,
                        "has_frame": frame_file.exists(),
                        "frame_url": f"/static/ar_assets/frames/{character_name}_frame.png" if frame_file.exists() else None
                    }
                    
                    characters.append(character_data)
                    logger.info(f"AR Character: {character_name} (webm: {webm_file.exists()}, frame: {frame_file.exists()})")
        
        # FIXED: Create sample data with proper structure
        if not characters:
            sample_characters = [
                {
                    "name": "boy",
                    "display_name": "Boy",
                    "thumbnail": "/static/ar_assets/thumbnail/boy.png",
                    "has_animation": True,
                    "animation_url": "/static/ar_assets/boy.webm",
                    "has_frame": True,
                    "frame_url": "/static/ar_assets/frames/boy_frame.png"
                },
                {
                    "name": "ghost", 
                    "display_name": "Ghost",
                    "thumbnail": "/static/ar_assets/thumbnail/ghost.png",
                    "has_animation": True,
                    "animation_url": "/static/ar_assets/ghost.webm",
                    "has_frame": True,
                    "frame_url": "/static/ar_assets/frames/ghost_frame.png"
                },
                {
                    "name": "superman",
                    "display_name": "Superman",
                    "thumbnail": "/static/ar_assets/thumbnail/superman.png", 
                    "has_animation": True,
                    "animation_url": "/static/ar_assets/superman.webm",
                    "has_frame": True,
                    "frame_url": "/static/ar_assets/frames/superman_frame.png"
                }
            ]
            characters = sample_characters
        
        return JSONResponse({
            "success": True,
            "characters": characters,
            "count": len(characters)
        })
    
    except Exception as e:
        logger.error(f"Error listing AR characters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR characters"
        )

@router.post("/photo")
async def create_ar_photo(
    photo: UploadFile = File(...),
    character_name: str = Form(...),  # FIXED: Changed from overlay_name to character_name
    current_user = Depends(get_current_user_ar)
):
    """FIXED: Process captured photo with AR overlay"""
    temp_files = []
    
    try:
        # Check credits before proceeding (admin bypasses)
        if current_user.get("role") != "admin":
            if current_user.get("credit_balance", 0) < 1:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Insufficient credits. Please make a payment."
                )
        
        # Validate uploaded photo
        validate_file(photo)

        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Ensure directories exist
        create_ar_directories()
        
        # Save uploaded photo
        photo_filename = generate_unique_filename(photo.filename, "ar_capture_")
        photo_path = ARConfig.UPLOAD_DIR / photo_filename
        await save_uploaded_file(photo, photo_path)
        temp_files.append(photo_path)
        
        # --- Multi-layer overlay processing ---

        # 1. Define paths for all overlays and results
        character_frame_filename = f"{character_name.strip().lower()}_frame.png"

        character_frame_path = ARConfig.FRAME_DIR / character_frame_filename

        # --- DIAGNOSTIC LOGS ---
        logger.info("--- JUMBO FRAME DIAGNOSTIC ---")
        logger.info(f"Received character_name: '{character_name}'")
        logger.info(f"Attempting to build path for: '{character_frame_filename}'")
        logger.info(f"Full character_frame_path: '{character_frame_path}'")
        logger.info(f"Does a file exist at this path? -> {character_frame_path.exists()}")
        logger.info("--- END DIAGNOSTIC ---")

        generic_frame_path = Path("static/images/frame1.png")

        username = current_user["username"]
        user_ar_dir = ARConfig.AR_RESULTS_DIR / username
        user_ar_dir.mkdir(parents=True, exist_ok=True)

        # Define paths for intermediate and final results
        intermediate_filename = f"intermediate_{unique_id}.png"
        intermediate_path = ARConfig.UPLOAD_DIR / intermediate_filename
        temp_files.append(intermediate_path)

        result_filename = f"{username}_ar_{character_name}_{timestamp}_{unique_id}.png"
        result_path = user_ar_dir / result_filename

        # 2. Step 1: Apply character-specific frame to the original photo
        logger.info(f"Step 1: Applying character frame '{character_frame_path}' to '{photo_path}'")
        intermediate_result_path = apply_ar_overlay(
            photo_path, character_frame_path, intermediate_path
        )

        # 3. Step 2: Apply generic frame on top of the intermediate result
        logger.info(f"Step 2: Applying generic frame '{generic_frame_path}' to '{intermediate_result_path}'")
        final_result_path = apply_ar_overlay(
            intermediate_result_path, generic_frame_path, result_path
        )

        
        # FIXED: Verify result file exists and has content
        if not final_result_path.exists() or final_result_path.stat().st_size == 0:
            raise ARPhotoError("Result file was not created or is empty")
        
        logger.info(f"AR photo created: {final_result_path} (size: {final_result_path.stat().st_size} bytes)")

        # Generate download URL and QR code with user-specific path
        download_url = f"/static/ar_results/{username}/{result_filename}"

        full_download_url = f"{ARConfig.DOMAIN_URL}{download_url}"
        qr_code_data = generate_qr_code(full_download_url)
        
        # Save to database, deduct credit, and record in 'photos' table
        try:
            from main import auth_service
            with auth_service.db_manager.get_connection() as conn:
                credits_used = 0
                # Deduct credit (if not admin)
                if current_user.get("role") != "admin":
                    credits_used = 1
                    conn.execute(
                        "UPDATE users SET credit_balance = credit_balance - 1 WHERE id = ?",
                        (current_user["id"],)
                    )

                # Record photo in the modern 'photos' table
                conn.execute("""
                    INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, credits_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (current_user["id"], result_filename, "ar_photo", character_name, str(final_result_path), credits_used))
                conn.commit()
        except ImportError:

            logger.warning("Database service not available")
        
        response_data = {
            "success": True,
            "message": "AR photo berhasil dibuat",
            "data": {
                "result_url": download_url,
                "result_filename": result_filename,
                "qr_code": qr_code_data,
                "download_url": full_download_url,
                "character_used": character_name,
                "overlay_used": f"{character_frame_filename} + {generic_frame_path.name}",

                "processing_time": datetime.now().isoformat(),
                "file_size": final_result_path.stat().st_size
            }
        }
        
        logger.info(f"AR photo response: {response_data}")
        return JSONResponse(response_data)
    
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error in AR photo creation: {e}\n{tb_str}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Terjadi kesalahan dalam pembuatan AR photo: {str(e)}",
                "traceback": tb_str,
            }
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

def create_simple_frame(frame_path: Path):
    """Create a simple frame overlay if none exists"""
    try:
        # Create a simple border frame
        frame_size = (640, 480)  # Default camera resolution
        frame = np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)  # RGBA
        
        # Create border (white with transparency)
        border_width = 20
        frame[:border_width, :] = [255, 255, 255, 200]  # Top border
        frame[-border_width:, :] = [255, 255, 255, 200]  # Bottom border
        frame[:, :border_width] = [255, 255, 255, 200]  # Left border
        frame[:, -border_width:] = [255, 255, 255, 200]  # Right border
        
        # Save frame
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_path), frame)
        logger.info(f"Created simple frame: {frame_path}")
        
    except Exception as e:
        logger.error(f"Failed to create simple frame: {e}")

@router.get("/overlays")
async def list_ar_overlays():
    """FIXED: List available AR overlay frames"""
    try:
        overlays = []
        ARConfig.FRAME_DIR.mkdir(parents=True, exist_ok=True)
        
        if ARConfig.FRAME_DIR.exists():
            for file_path in ARConfig.FRAME_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ARConfig.ALLOWED_EXTENSIONS:
                    overlays.append({
                        "name": file_path.name,
                        "path": f"/static/ar_assets/frames/{file_path.name}",
                        "preview": f"/static/ar_assets/frames/{file_path.name}"
                    })
        
        # Default frame if no overlays exist
        if not overlays:
            # Create default frame1.png
            default_frame_path = ARConfig.FRAME_DIR / "frame1.png"
            if not default_frame_path.exists():
                create_simple_frame(default_frame_path)
            
            overlays = [
                {"name": "frame1.png", "path": "/static/ar_assets/frames/frame1.png", "preview": "/static/ar_assets/frames/frame1.png"}
            ]
        
        return JSONResponse({
            "success": True,
            "overlays": overlays,
            "count": len(overlays)
        })
    
    except Exception as e:
        logger.error(f"Error listing AR overlays: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AR overlays"
        )

@router.get("/test")
async def ar_test():
    """Test AR endpoints"""
    return {
        "success": True,
        "message": "AR Photo module working!",
        "directories": {
            "ar_assets": str(ARConfig.AR_ASSETS_DIR),
            "thumbnails": str(ARConfig.THUMBNAIL_DIR),
            "frames": str(ARConfig.FRAME_DIR),
            "results": str(ARConfig.AR_RESULTS_DIR)
        }
    }
