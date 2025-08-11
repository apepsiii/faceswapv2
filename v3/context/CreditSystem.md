# 💳 CONTEXT: CREDIT SYSTEM & ENHANCED PHOTO GENERATION

## **Current State:**
- Manual 3-photo limit via localStorage counter
- No credit tracking in database  
- Payment → redirect to character (no credit addition)
- Photos saved in single folders: `static/results/`, `static/ar_results/`

## **Target State:**
- Database-driven credit system
- Payment success → auto add credits to user
- Photo generation → auto deduct credits
- User-specific folder structure
- Enhanced filename convention

## **Credit System Flow:**

### **1. Payment → Credit Addition:**
```python
# Enhanced QRIS callback
@app.get("/api/qris/status")
async def check_qris_status(order_id: str):
    status = core_api.transactions.status(order_id)
    
    if status.get("transaction_status") == "settlement":
        # Add credits to user
        with auth_service.db_manager.get_connection() as conn:
            # Find transaction
            cursor = conn.execute("SELECT user_id FROM transactions WHERE order_id = ?", (order_id,))
            transaction = cursor.fetchone()
            
            if transaction:
                user_id = transaction[0]
                credits_to_add = 3  # From settings table
                
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
    
    return status
```

### **2. Photo Generation → Credit Deduction:**
```python
# Credit checking middleware
async def check_user_credits(current_user = Depends(get_current_user)):
    if current_user["credit_balance"] < 1:
        raise HTTPException(
            status_code=402, 
            detail="Insufficient credits. Please make a payment."
        )
    return current_user

# Enhanced photo generation
@app.post("/api/swap")
async def swap_faces_api(
    template_name: str = Form(...),
    webcam: UploadFile = File(...),
    current_user = Depends(check_user_credits)  # NEW
):
    # Generate filename dengan username
    username = current_user["username"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    result_filename = f"{username}_{timestamp}_{unique_id}.png"
    
    # Create user-specific folder
    user_result_dir = Config.RESULT_DIR / username
    user_result_dir.mkdir(parents=True, exist_ok=True)
    result_path = user_result_dir / result_filename
    
    # Process face swap (existing logic)
    swap_result_path = swap_faces(source_path, template_path, result_path)
    
    # Deduct credit dan record photo
    with auth_service.db_manager.get_connection() as conn:
        # Deduct credit
        conn.execute(
            "UPDATE users SET credit_balance = credit_balance - 1 WHERE id = ?",
            (current_user["id"],)
        )
        
        # Record photo
        conn.execute("""
            INSERT INTO photos (user_id, filename, photo_type, template_name, file_path, credits_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (current_user["id"], result_filename, "face_swap", template_name, str(result_path), 1))
        
        conn.commit()
    
    return {
        "success": True,
        "data": {
            "result_url": f"/static/results/{username}/{result_filename}",
            "result_filename": result_filename,
            "credits_remaining": current_user["credit_balance"] - 1
        }
    }
```

## **File Organization:**
```
static/
├── results/
│   ├── cbt/
│   │   ├── cbt_20250811143025_a1b2c3d4.png
│   │   └── cbt_20250811143530_x9y8z7w6.png
│   ├── bsd/
│   │   └── bsd_20250811144015_m5n6o7p8.png
│   └── slo/
├── ar_results/
│   ├── cbt/
│   ├── bsd/
│   └── slo/
└── templates/ (unchanged)
```

## **Enhanced Payment Flow:**
```python
@app.get("/api/qris/token")  
async def generate_qris_token(user_id: int = Query(...)):
    order_id = f"ORDER-{uuid.uuid4().hex[:12]}"
    
    # Record transaction
    with auth_service.db_manager.get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions (user_id, order_id, amount, credits_added, status)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, order_id, 5000, 3, "pending"))
        conn.commit()
    
    # Generate QRIS (existing logic)
    payload = {
        "transaction_details": {
            "order_id": order_id,
            "gross_amount": 5000
        }
    }
    
    result = core_api.charge(payload)
    return {"success": True, "qris_url": qris_url, "order_id": order_id}
```

## **Credit Management APIs:**
```python
@app.get("/api/user/credits")
async def get_user_credits(current_user = Depends(get_current_user)):
    return {
        "success": True,
        "credits": current_user["credit_balance"],
        "username": current_user["username"]
    }

@app.post("/api/admin/users/{user_id}/add-credits")  
async def add_credits(user_id: int, credits: int, admin_user = Depends(admin_required)):
    with auth_service.db_manager.get_connection() as conn:
        conn.execute(
            "UPDATE users SET credit_balance = credit_balance + ? WHERE id = ?",
            (credits, user_id)
        )
        conn.commit()
    
    return {"success": True, "message": f"Added {credits} credits"}
```

## **Frontend Integration:**
- Payment page pass `user_id` to QRIS generation
- Photo pages show credit balance  
- Auto-redirect to payment jika credit habis
- Real-time credit updates

## **Implementation Steps:**
1. Update photo generation endpoints dengan credit checking
2. Implement user-specific folder creation
3. Enhance payment flow dengan transaction recording
4. Update frontend untuk credit display
5. Test complete flow: payment → credit addition → photo → credit deduction