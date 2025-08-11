# 🚀 CONTEXT: IMPLEMENTATION ROADMAP & NEXT STEPS

## **IMMEDIATE ACTION PLAN**

### **🎯 PHASE 1: FOUNDATION (Priority: CRITICAL)**
**Estimated Time: 2-3 days**

#### **Step 1.1: Database Migration (Day 1)**
```bash
# Tasks to complete:
1. Backup existing database
2. Run schema alterations
3. Create new tables
4. Insert default data
5. Test migrations
```

**Files to modify:**
- `main.py` - Add migration functions
- `auth.py` - Update for role-based auth
- Create `migration.py` - Database migration script

#### **Step 1.2: Enhanced Authentication (Day 2)**
```bash
# Tasks to complete:
1. Update AuthService class
2. Add role-based middleware
3. Update login endpoint
4. Test role redirects
5. Update frontend login page
```

**Files to create/modify:**
- `auth.py` - Enhanced with roles
- `pages/login.html` - Role-based redirect
- `main.py` - Role middleware

#### **Step 1.3: Foundation Testing (Day 3)**
```bash
# Tasks to complete:
1. Test admin login → dashboard_admin
2. Test user login → dashboard  
3. Verify role permissions
4. Test existing functionality
5. Fix any breaking changes
```

---

### **🎯 PHASE 2: CREDIT SYSTEM (Priority: HIGH)**
**Estimated Time: 3-4 days**

#### **Step 2.1: Credit Management Backend (Day 4-5)**
```bash
# Tasks to complete:
1. Update QRIS payment flow
2. Add credit checking middleware
3. Update photo generation endpoints
4. Create user-specific folders
5. Test payment → credit addition
```

**Files to modify:**
- `main.py` - Enhanced photo endpoints
- Payment endpoints - Credit addition
- File structure - User folders

#### **Step 2.2: Enhanced Photo Generation (Day 6-7)**
```bash
# Tasks to complete:
1. Implement filename convention
2. Credit deduction logic
3. User folder creation
4. Update both face swap and AR
5. Test complete flow
```

---

### **🎯 PHASE 3: ADMIN DASHBOARD (Priority: HIGH)**  
**Estimated Time: 4-5 days**

#### **Step 3.1: Dashboard Backend (Day 8-9)**
```bash
# Tasks to complete:
1. Create analytics APIs
2. Optimize dashboard queries
3. Test data accuracy
4. Add caching if needed
5. Performance optimization
```

**Files to create:**
- `pages/dashboard_admin.html` - Admin dashboard
- Analytics APIs in `main.py`

#### **Step 3.2: Dashboard Frontend (Day 10-12)**
```bash
# Tasks to complete:
1. Create dashboard HTML/CSS
2. Implement Chart.js integration
3. Add real-time updates
4. Mobile responsive design
5. Test all charts and metrics
```

---

## **🔧 TECHNICAL IMPLEMENTATION CHECKLIST**

### **Database Migration Checklist:**
```sql
-- ✅ Backup current database
-- ✅ ALTER users table (add role, credit_balance)
-- ✅ CREATE transactions table
-- ✅ CREATE photos table  
-- ✅ CREATE settings table
-- ✅ CREATE indexes for performance
-- ✅ INSERT default admin user
-- ✅ INSERT sample users (cbt, bsd, slo, mgl, sdo, plp)
-- ✅ INSERT default settings
-- ✅ Test all queries work
```

### **Authentication Enhancement Checklist:**
```python
# ✅ Update AuthService.login_user() - add role/credit to response
# ✅ Update AuthService.get_user_by_token() - include role/credit
# ✅ Create admin_required() middleware
# ✅ Create user_required() middleware  
# ✅ Update /api/login endpoint - role-based redirect
# ✅ Update /api/me endpoint - include role/credit
# ✅ Update login.html - handle role-based redirect
# ✅ Test admin vs user access
```

### **Credit System Checklist:**
```python
# ✅ Create check_user_credits() middleware
# ✅ Update /api/qris/token - include user_id
# ✅ Update /api/qris/status - auto-add credits on settlement
# ✅ Update /api/swap - credit checking + deduction  
# ✅ Update /api/ar/photo - credit checking + deduction
# ✅ Create user-specific folders structure
# ✅ Implement filename convention: {username}_{timestamp}_{random}
# ✅ Test payment → photo → credit flow
```

### **Admin Dashboard Checklist:**
```python
# ✅ Create /api/admin/dashboard/stats
# ✅ Create /api/admin/dashboard/sales-chart
# ✅ Create /api/admin/dashboard/usage-chart  
# ✅ Create /api/admin/dashboard/user-photos-pie
# ✅ Create dashboard_admin.html with responsive design
# ✅ Implement Chart.js visualizations
# ✅ Add real-time updates (30-second refresh)
# ✅ Test all metrics accuracy
```

---

## **📁 FILE STRUCTURE AFTER IMPLEMENTATION**

```
project/
├── main.py                 # Enhanced with all new endpoints
├── auth.py                 # Enhanced with role-based auth
├── migration.py            # Database migration script (NEW)
├── face_swap.db           # Enhanced database schema
├── pages/
│   ├── login.html         # Enhanced with role-based redirect
│   ├── dashboard.html     # Existing user dashboard
│   ├── dashboard_admin.html # NEW - Admin dashboard
│   ├── user_management.html # NEW - User CRUD interface
│   ├── settings_admin.html  # NEW - Admin settings
│   └── [other existing pages]
├── static/
│   ├── results/
│   │   ├── cbt/           # NEW - User-specific folders
│   │   ├── bsd/           # NEW
│   │   └── slo/           # NEW  
│   ├── ar_results/
│   │   ├── cbt/           # NEW - User-specific folders
│   │   ├── bsd/           # NEW
│   │   └── slo/           # NEW
│   └── [other existing static files]
```

---

## **🧪 TESTING STRATEGY**

### **Phase 1 Testing:**
```bash
1. Database migration successful
2. Admin login → /dashboard_admin
3. User login → /dashboard  
4. Role permissions working
5. Existing functionality not broken
```

### **Phase 2 Testing:**
```bash
1. Payment → credit addition works
2. Photo generation → credit deduction works
3. User-specific folders created
4. Filename convention correct
5. Credit exhaustion → payment redirect
```

### **Phase 3 Testing:**
```bash
1. Dashboard metrics accurate
2. Charts display correct data
3. Real-time updates working
4. Mobile responsiveness
5. Performance acceptable
```

---

## **🚨 POTENTIAL ISSUES & SOLUTIONS**

### **Migration Issues:**
```bash
Problem: Existing data migration
Solution: Create backup before migration, test on copy first

Problem: Foreign key constraints
Solution: Disable FK checks during migration, re-enable after

Problem: Large database migration time
Solution: Run migration during low-traffic hours
```

### **Performance Issues:**
```bash
Problem: Dashboard queries slow
Solution: Add proper indexes, implement caching

Problem: Too many API calls
Solution: Combine related data in single endpoints

Problem: Large user folders
Solution: Implement file cleanup policies
```

### **Authentication Issues:**
```bash
Problem: Existing sessions invalid after role update
Solution: Force re-login after migration

Problem: Role permission conflicts
Solution: Clear role hierarchy and test thoroughly
```

---

## **📋 DEVELOPMENT COMMANDS**

### **Database Migration Commands:**
```bash
# Backup database
cp face_swap.db face_swap_backup_$(date +%Y%m%d).db

# Run migration
python migration.py

# Verify migration
sqlite3 face_swap.db ".schema"
sqlite3 face_swap.db "SELECT COUNT(*) FROM users WHERE role='admin';"
```

### **Testing Commands:**
```bash
# Test admin login
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Test dashboard API
curl -H "Authorization: Bearer <token>" \
  http://localhost:5000/api/admin/dashboard/stats

# Test credit system
curl -X POST http://localhost:5000/api/swap \
  -H "Authorization: Bearer <user_token>" \
  -F "template_name=superhero.jpg" \
  -F "webcam=@test_image.jpg"
```

### **Development Server Commands:**
```bash
# Development with auto-reload
python main.py

# Production-like testing
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 1
```

---

## **📊 SUCCESS METRICS**

### **Phase 1 Success Criteria:**
- ✅ Admin dapat login dan access dashboard admin
- ✅ User dapat login dan access dashboard user  
- ✅ Role permissions berfungsi dengan benar
- ✅ Database migration berhasil tanpa data loss
- ✅ Existing functionality masih bekerja

### **Phase 2 Success Criteria:**
- ✅ Payment QRIS → auto add 3 credits ke user
- ✅ Generate foto → auto deduct 1 credit
- ✅ Credit habis → redirect ke payment
- ✅ File tersimpan di folder user: `/static/results/{username}/`
- ✅ Filename format: `{username}_{timestamp}_{random}.png`

### **Phase 3 Success Criteria:**
- ✅ Dashboard menampilkan metrics real-time
- ✅ Charts menampilkan data akurat
- ✅ Mobile responsive (< 768px)
- ✅ Page load time < 3 detik
- ✅ Auto-refresh setiap 30 detik

---

## **🔄 CONTINUATION CONTEXT FOR NEW CHAT**

### **When Starting New Chat, Use This Prompt:**

```
I'm continuing development of a Multi-Role Photobooth System. Current status:

COMPLETED:
- Basic face swap + AR photo system with SQLite
- QRIS payment integration  
- Single-role authentication

WORKING ON: [Current phase from above]

TECH STACK: FastAPI + SQLite + HTML/CSS/JS + Midtrans QRIS

KEY REQUIREMENTS:
- Admin role: Dashboard with analytics, user management, settings
- User role: Basic dashboard, photo generation with credit system
- Credit system: Rp 5,000 = 3 credits = 3 photos
- File organization: User-specific folders with naming convention {username}_{timestamp}_{random}
- Database: Enhanced SQLite with users, transactions, photos, settings tables

CURRENT TASK: [Specify which phase/step you're working on]

Please help me implement [specific task] based on the requirements above.
```

### **File-Specific Contexts Available:**
1. **Database Context** - Schema, migrations, queries
2. **Auth Context** - Role-based authentication system  
3. **Credit Context** - Payment integration, credit management
4. **Dashboard Context** - Admin analytics interface
5. **User Management Context** - CRUD operations, user analytics

### **Key URLs for Reference:**
- Login: `/login` → Role-based redirect
- Admin Dashboard: `/dashboard_admin` 
- User Dashboard: `/dashboard`
- User Management: `/user_management` (admin only)
- Settings: `/settings_admin` (admin only)

---

## **📞 READY FOR IMPLEMENTATION**

**Current status: READY TO START PHASE 1**

**Next immediate task: Database Schema Migration**

**Files to create first:**
1. `migration.py` - Database migration script
2. Enhanced `auth.py` - Role-based authentication
3. Updated `main.py` - New endpoints and middleware

**Estimated completion: 15-20 working days for full system**

**MVP (Phases 1-3): 9-12 working days**

---

**🚀 Ready to begin implementation whenever you're ready to continue!**