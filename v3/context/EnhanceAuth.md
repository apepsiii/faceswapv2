# 🔐 CONTEXT: ENHANCED AUTHENTICATION SYSTEM

## **Current State:**
- Basic JWT authentication working
- Single role system (all users equal)
- Login → dashboard → character flow
- `AuthService` class in `auth.py`

## **Target State:**
- Role-based authentication (admin/user)
- Login redirects based on role:
  - Admin → `/dashboard_admin`
  - User → `/dashboard` (existing)
- Credit balance included in user session
- Enhanced session management

## **Key Changes Needed:**

### **1. Enhanced AuthService (`auth.py`):**
```python
class AuthService:
    def login_user(self, login_data: UserLogin) -> dict:
        # Enhanced response dengan role dan credit
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user_id,
                "username": username,
                "role": role,  # NEW
                "credit_balance": credit_balance  # NEW
            },
            "redirect_url": "/dashboard_admin" if role == "admin" else "/dashboard"  # NEW
        }
    
    def get_user_by_token(self, token: str) -> dict:
        # Include role dan credit dalam response
        return {
            "id": user[0],
            "username": user[1], 
            "role": user[2],  # NEW
            "credit_balance": user[3],  # NEW
            "created_at": user[4],
            "last_login": user[5]
        }
```

### **2. Role-Based Middleware:**
```python
async def admin_required(current_user = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

async def user_required(current_user = Depends(get_current_user)):
    if current_user.get("role") not in ["admin", "user"]:
        raise HTTPException(status_code=403, detail="User access required") 
    return current_user
```

### **3. Enhanced Login Endpoint (`main.py`):**
```python
@app.post("/api/login")
async def login(login_data: UserLogin):
    result = auth_service.login_user(login_data)
    # Frontend akan redirect based on result.redirect_url
    return result
```

### **4. Frontend Login Update (`pages/login.html`):**
```javascript
// Enhanced login handler
const data = await response.json();
if (data.success) {
    localStorage.setItem('token', data.token);
    localStorage.setItem('user', JSON.stringify(data.user));
    
    // Role-based redirect
    window.location.replace(data.redirect_url);
}
```

## **New API Endpoints:**
```python
# Enhanced user info
@app.get("/api/me")
async def get_me(current_user = Depends(get_current_user)):
    return {"success": True, "user": current_user}

# Admin-only endpoints
@app.get("/api/admin/dashboard")
async def admin_dashboard(admin_user = Depends(admin_required)):
    return admin_dashboard_data()

# User credit info
@app.get("/api/user/credits")
async def get_credits(current_user = Depends(user_required)):
    return {"credits": current_user["credit_balance"]}
```

## **Frontend Pages:**
- `pages/login.html` - Enhanced dengan role-based redirect
- `pages/dashboard.html` - Existing user dashboard  
- `pages/dashboard_admin.html` - NEW admin dashboard
- Role-based navigation components

## **Implementation Priority:**
1. Update database schema dengan role dan credit_balance
2. Enhance AuthService dengan role checking
3. Update login endpoint dan frontend
4. Create admin dashboard foundation
5. Implement role-based middleware

## **Testing Checklist:**
- [ ] Admin login → redirect to `/dashboard_admin`  
- [ ] User login → redirect to `/dashboard`
- [ ] Role validation working
- [ ] Credit balance displayed correctly
- [ ] Existing functionality not broken