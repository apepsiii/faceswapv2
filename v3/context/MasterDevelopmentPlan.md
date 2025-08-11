# 🎯 MASTER DEVELOPMENT PLAN
## Multi-Role Photobooth System with Admin Dashboard

### 📊 **PROJECT OVERVIEW**
- **Current System**: Single-role face swap + AR photo dengan SQLite
- **Target System**: Multi-role (Admin/User) dengan credit system, analytics dashboard, dan user management
- **Tech Stack**: FastAPI + SQLite + HTML/CSS/JS + Midtrans QRIS
- **Business Model**: Rp 5.000 = 3 credits = 3 foto, per-user tracking

---

## 🏗️ **DEVELOPMENT PHASES**

### **PHASE 1: DATABASE & AUTH FOUNDATION** ⏱️ 2-3 hari
**Priority: CRITICAL - Foundation untuk semua fitur**

#### 1.1 Database Schema Enhancement
- [ ] Update `users` table tambah `role` dan `credit_balance`
- [ ] Create `transactions` table untuk payment tracking
- [ ] Create `photos` table replace `face_swap_history`  
- [ ] Create `settings` table untuk dynamic config
- [ ] Create `user_sessions` table untuk session management
- [ ] Add indexes untuk dashboard performance
- [ ] Insert default admin user dan sample users (cbt, bsd, slo, mgl, sdo, plp)

#### 1.2 Enhanced Authentication System
- [ ] Update `AuthService` class dengan role-based authentication
- [ ] Implement role-based middleware (`@admin_required`, `@user_required`)
- [ ] Update login endpoint dengan role-based redirect
- [ ] Enhanced session management dengan credit tracking
- [ ] API endpoint `/api/me` return role dan credit_balance

**Deliverables**: Enhanced database, role-based auth, admin/user separation

---

### **PHASE 2: CREDIT SYSTEM & ENHANCED PHOTO** ⏱️ 3-4 hari  
**Priority: HIGH - Core business logic**

#### 2.1 Credit Management System
- [ ] Credit checking middleware untuk photo endpoints
- [ ] Auto-deduct credit saat generate foto
- [ ] Redirect ke payment jika credit habis
- [ ] API endpoints: `/api/user/credits/*`

#### 2.2 Enhanced Photo Generation
- [ ] Update filename format: `{username}_{timestamp}_{random}.png`
- [ ] Create user-specific folders: `static/results/{username}/`
- [ ] Update `/api/swap` dengan credit checking + user folder
- [ ] Update `/api/ar/photo` dengan credit checking + user folder
- [ ] Enhanced photo metadata tracking

#### 2.3 Enhanced Payment System
- [ ] Update `/api/qris/token` dengan user_id tracking
- [ ] Enhanced `/api/qris/status` dengan auto-credit addition
- [ ] Transaction recording ke database
- [ ] Payment success → auto add credits

**Deliverables**: Working credit system, user-specific photo storage, enhanced payment

---

### **PHASE 3: ADMIN DASHBOARD** ⏱️ 4-5 hari
**Priority: HIGH - Main deliverable**

#### 3.1 Dashboard Analytics Backend
- [ ] API `/api/admin/dashboard/stats` - basic counts
- [ ] API `/api/admin/dashboard/sales-chart` - revenue trends  
- [ ] API `/api/admin/dashboard/usage-chart` - photo usage trends
- [ ] API `/api/admin/dashboard/user-photos-pie` - user distribution
- [ ] Optimized queries untuk real-time performance

#### 3.2 Admin Dashboard Frontend  
- [ ] Create `pages/dashboard_admin.html` sesuai design
- [ ] Implement responsive charts (Chart.js atau similar)
- [ ] Real-time metrics dengan auto-refresh
- [ ] Mobile-responsive design
- [ ] Role-based navigation menu

#### 3.3 Dashboard Components
- [ ] Top cards: Total users, photos, revenue
- [ ] Sales line chart dengan filter period
- [ ] Usage bar chart face swap vs AR
- [ ] User pie chart distribution  
- [ ] Today's summary section

**Deliverables**: Complete admin dashboard dengan real-time analytics

---

### **PHASE 4: USER MANAGEMENT** ⏱️ 2-3 hari
**Priority: MEDIUM - Admin tools**

#### 4.1 User Management Backend
- [ ] API `/api/admin/users` - CRUD operations
- [ ] API `/api/admin/users/{id}/reset-credits` 
- [ ] API `/api/admin/users/{id}/details` - detailed stats
- [ ] User activity tracking dan reporting

#### 4.2 User Management Frontend
- [ ] Create `pages/user_management.html`
- [ ] User list table dengan stats
- [ ] Add/Edit/Delete user forms
- [ ] Credit management tools
- [ ] User performance analytics

**Deliverables**: Complete user management system

---

### **PHASE 5: SETTINGS & CONFIGURATION** ⏱️ 1-2 hari  
**Priority: MEDIUM - Admin flexibility**

#### 5.1 Dynamic Settings System
- [ ] API `/api/admin/settings` - get/update settings
- [ ] Dynamic pricing system
- [ ] Business configuration options
- [ ] System maintenance tools

#### 5.2 Settings Frontend  
- [ ] Create `pages/settings_admin.html`
- [ ] Pricing configuration form
- [ ] System settings panel
- [ ] Maintenance tools interface

**Deliverables**: Dynamic configuration system

---

### **PHASE 6: EXPORT & REPORTING** ⏱️ 2-3 hari
**Priority: LOW - Business intelligence**

#### 6.1 Export System
- [ ] API `/api/admin/export/*` - Excel/CSV exports
- [ ] Transaction reports dengan date filtering
- [ ] Photo reports per user
- [ ] User summary reports
- [ ] Financial summary reports

#### 6.2 Advanced Analytics
- [ ] Revenue forecasting
- [ ] User behavior analysis  
- [ ] Peak time analysis
- [ ] Performance optimization reports

**Deliverables**: Complete reporting system

---

### **PHASE 7: OPTIMIZATION & DEPLOYMENT** ⏱️ 2-3 hari
**Priority: LOW - Production readiness**

#### 7.1 Performance Optimization
- [ ] Database query optimization
- [ ] File storage optimization  
- [ ] Caching implementation
- [ ] Auto-backup system

#### 7.2 Production Deployment
- [ ] Environment configuration
- [ ] Security hardening
- [ ] Monitoring setup
- [ ] Documentation

**Deliverables**: Production-ready system

---

## 🎯 **CURRENT STATUS & NEXT STEPS**

### ✅ **Completed:**
- Basic face swap + AR photo system
- SQLite database foundation
- QRIS payment integration
- User session management (3 foto limit)

### 🔄 **Currently Working On:**
- Planning dan persiapan Phase 1

### ⏭️ **Next Immediate Tasks:**
1. **Database Schema Update** (Phase 1.1)
2. **Enhanced Authentication** (Phase 1.2)  
3. **Credit System Foundation** (Phase 2.1)

### 📅 **Timeline Estimate:**
- **Total Development**: 15-20 hari kerja
- **MVP Ready**: Phase 1-3 (9-12 hari)
- **Full System**: All phases (15-20 hari)

---

## 🔧 **TECHNICAL DECISIONS**

### **Database**: SQLite (optimal untuk single-location photobooth)
### **Authentication**: Role-based dengan JWT tokens  
### **File Storage**: User-specific folders dengan naming convention
### **Payment**: Midtrans QRIS dengan credit system
### **Frontend**: Vanilla HTML/CSS/JS (lightweight, fast)
### **Analytics**: Real-time dengan optimized SQL queries

---

## 📝 **DEVELOPMENT NOTES**

### **Code Structure:**
```
main.py - Core FastAPI application
auth.py - Authentication service (existing)
pages/ - Frontend HTML files
static/ - Static assets dan user folders
```

### **Key Features:**
- Role-based access (admin/user)
- Credit-based photo generation
- User-specific file organization  
- Real-time analytics dashboard
- Dynamic pricing configuration
- Comprehensive reporting

### **Business Logic:**
- 1 pembayaran = 3 credits = 3 foto
- Filename format: `{username}_{timestamp}_{random}`
- Auto-redirect ke payment saat credit habis
- Admin full control + analytics
- User hanya akses foto + dashboard basic


Planning

🗺️ Master Development Plan - Roadmap lengkap 7 fase pengembangan
🗃️ Database Context - Schema enhancement & migration plan
🔐 Auth Context - Role-based authentication system
💳 Credit Context - Payment integration & credit management
📊 Dashboard Context - Admin analytics interface
👥 User Management Context - CRUD operations & user analytics
🚀 Implementation Roadmap - Next steps & continuation guide