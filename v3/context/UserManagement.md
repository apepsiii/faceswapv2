# 👥 CONTEXT: USER MANAGEMENT SYSTEM

## **Current State:**
- Manual user creation via database
- No admin interface untuk user management
- Basic user info di database (username, password, role)
- No user analytics atau performance tracking

## **Target State:**
- Complete CRUD interface untuk user management
- User performance analytics
- Credit management tools
- Bulk operations (reset credits, deactivate users)
- User activity monitoring

## **User Management Features Required:**

### **1. User List Interface:**
```javascript
// User data structure
{
    "users": [
        {
            "id": 1,
            "username": "cbt",
            "role": "user", 
            "credit_balance": 5,
            "total_photos": 45,
            "total_spent": 75000,
            "last_photo": "2025-01-16T10:30:00",
            "last_login": "2025-01-16T09:15:00",
            "created_at": "2024-12-01T00:00:00",
            "is_active": true,
            "performance": {
                "face_swap_count": 25,
                "ar_photo_count": 20,
                "avg_photos_per_session": 2.8,
                "revenue_generated": 75000
            }
        }
    ]
}
```

### **2. Backend APIs:**
```python
# User CRUD Operations
@app.get("/api/admin/users")
async def list_users(admin_user = Depends(admin_required)):
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

@app.post("/api/admin/users")
async def create_user(user_data: dict, admin_user = Depends(admin_required)):
    # Create new user implementation
    
@app.put("/api/admin/users/{user_id}")
async def update_user(user_id: int, user_data: dict, admin_user = Depends(admin_required)):
    # Update user implementation
    
@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, admin_user = Depends(admin_required)):
    # Delete user implementation (soft delete)
    
@app.post("/api/admin/users/{user_id}/reset-credits")
async def reset_user_credits(user_id: int, credits: int, admin_user = Depends(admin_required)):
    with auth_service.db_manager.get_connection() as conn:
        conn.execute(
            "UPDATE users SET credit_balance = ? WHERE id = ?",
            (credits, user_id)
        )
        conn.commit()
    
    return {"success": True, "message": f"Credits reset to {credits}"}

@app.get("/api/admin/users/{user_id}/details")
async def get_user_details(user_id: int, admin_user = Depends(admin_required)):
    # Detailed user analytics dengan photo history, transaction history, etc
```

### **3. Frontend Interface (`pages/user_management.html`):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>User Management - Admin Panel</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <!-- Include sidebar from dashboard_admin -->
    <div class="main-content">
        <!-- Header with Add User Button -->
        <div class="page-header">
            <h1>User Management</h1>
            <button class="btn btn-primary" id="addUserBtn">
                <i class="fas fa-plus"></i> Add New User
            </button>
        </div>
        
        <!-- Filters and Search -->
        <div class="filters-section">
            <div class="search-box">
                <input type="text" placeholder="Search users..." id="searchInput">
                <i class="fas fa-search"></i>
            </div>
            <select id="roleFilter">
                <option value="">All Roles</option>
                <option value="user">User</option>
                <option value="admin">Admin</option>
            </select>
            <select id="statusFilter">
                <option value="">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
            </select>
        </div>
        
        <!-- Users Table -->
        <div class="table-container">
            <table class="users-table">
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Credits</th>
                        <th>Total Photos</th>
                        <th>Revenue Generated</th>
                        <th>Last Activity</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="usersTableBody">
                    <!-- Users will be loaded here -->
                </tbody>
            </table>
        </div>
        
        <!-- Pagination -->
        <div class="pagination">
            <button id="prevPage">Previous</button>
            <span id="pageInfo">Page 1 of 1</span>
            <button id="nextPage">Next</button>
        </div>
    </div>
    
    <!-- Add/Edit User Modal -->
    <div class="modal" id="userModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modalTitle">Add New User</h2>
                <button class="close-btn">&times;</button>
            </div>
            <form id="userForm">
                <div class="form-group">
                    <label>Username:</label>
                    <input type="text" id="username" required>
                </div>
                <div class="form-group">
                    <label>Password:</label>
                    <input type="password" id="password" required>
                </div>
                <div class="form-group">
                    <label>Role:</label>
                    <select id="role">
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Initial Credits:</label>
                    <input type="number" id="credits" value="0" min="0">
                </div>
                <div class="form-actions">
                    <button type="button" class="btn btn-secondary" id="cancelBtn">Cancel</button>
                    <button type="submit" class="btn btn-primary">Save User</button>
                </div>
            </form>
        </div>
    </div>
    
    <!-- Credit Management Modal -->
    <div class="modal" id="creditModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Manage Credits - <span id="creditUsername"></span></h2>
                <button class="close-btn">&times;</button>
            </div>
            <div class="credit-info">
                <div class="current-credits">
                    Current Credits: <span id="currentCredits">0</span>
                </div>
            </div>
            <form id="creditForm">
                <div class="form-group">
                    <label>Action:</label>
                    <select id="creditAction">
                        <option value="set">Set Credits To</option>
                        <option value="add">Add Credits</option>
                        <option value="subtract">Subtract Credits</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Amount:</label>
                    <input type="number" id="creditAmount" min="0" required>
                </div>
                <div class="form-group">
                    <label>Reason (Optional):</label>
                    <textarea id="creditReason" placeholder="Reason for credit adjustment..."></textarea>
                </div>
                <div class="form-actions">
                    <button type="button" class="btn btn-secondary">Cancel</button>
                    <button type="submit" class="btn btn-primary">Update Credits</button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>
```

### **4. JavaScript Implementation:**
```javascript
class UserManagement {
    constructor() {
        this.users = [];
        this.currentPage = 1;
        this.usersPerPage = 10;
        this.filters = {
            search: '',
            role: '',
            status: ''
        };
        this.init();
    }
    
    async init() {
        await this.loadUsers();
        this.setupEventListeners();
        this.renderUsers();
    }
    
    async loadUsers() {
        try {
            const response = await fetch('/api/admin/users', {
                headers: {'Authorization': `Bearer ${localStorage.getItem('token')}`}
            });
            const data = await response.json();
            
            if (data.success) {
                this.users = data.users;
            }
        } catch (error) {
            console.error('Failed to load users:', error);
        }
    }
    
    renderUsers() {
        const filteredUsers = this.applyFilters();
        const paginatedUsers = this.paginateUsers(filteredUsers);
        
        const tbody = document.getElementById('usersTableBody');
        tbody.innerHTML = '';
        
        paginatedUsers.forEach(user => {
            const row = this.createUserRow(user);
            tbody.appendChild(row);
        });
        
        this.updatePagination(filteredUsers.length);
    }
    
    createUserRow(user) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <div class="user-info">
                    <div class="username">${user.username}</div>
                    <div class="user-id">ID: ${user.id}</div>
                </div>
            </td>
            <td>
                <span class="credit-badge ${user.credit_balance > 0 ? 'positive' : 'zero'}">
                    ${user.credit_balance}
                </span>
            </td>
            <td>
                <div class="photo-stats">
                    <div>Total: ${user.total_photos}</div>
                    <div class="breakdown">
                        Face Swap: ${user.face_swap_count} | AR: ${user.ar_photo_count}
                    </div>
                </div>
            </td>
            <td class="revenue">
                ${this.formatCurrency(user.total_spent)}
            </td>
            <td>
                <div class="activity-info">
                    ${user.last_photo ? new Date(user.last_photo).toLocaleDateString() : 'Never'}
                </div>
            </td>
            <td>
                <span class="status-badge ${user.is_active ? 'active' : 'inactive'}">
                    ${user.is_active ? 'Active' : 'Inactive'}
                </span>
            </td>
            <td>
                <div class="action-buttons">
                    <button class="btn-icon" onclick="userManagement.editUser(${user.id})" title="Edit">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn-icon" onclick="userManagement.manageCredits(${user.id})" title="Manage Credits">
                        <i class="fas fa-coins"></i>
                    </button>
                    <button class="btn-icon" onclick="userManagement.viewDetails(${user.id})" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn-icon danger" onclick="userManagement.deleteUser(${user.id})" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        `;
        return row;
    }
    
    async manageCredits(userId) {
        const user = this.users.find(u => u.id === userId);
        if (!user) return;
        
        document.getElementById('creditUsername').textContent = user.username;
        document.getElementById('currentCredits').textContent = user.credit_balance;
        document.getElementById('creditModal').style.display = 'block';
        
        // Store current user for credit management
        this.currentCreditUser = user;
    }
    
    async updateUserCredits(action, amount, reason) {
        try {
            const endpoint = `/api/admin/users/${this.currentCreditUser.id}/credits`;
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    action: action,
                    amount: amount,
                    reason: reason
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                await this.loadUsers();
                this.renderUsers();
                document.getElementById('creditModal').style.display = 'none';
                this.showNotification('Credits updated successfully', 'success');
            }
        } catch (error) {
            console.error('Failed to update credits:', error);
            this.showNotification('Failed to update credits', 'error');
        }
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('id-ID', {
            style: 'currency',
            currency: 'IDR',
            minimumFractionDigits: 0
        }).format(amount);
    }
}

// Initialize user management
document.addEventListener('DOMContentLoaded', () => {
    window.userManagement = new UserManagement();
});
```

## **Key Features:**
- **Real-time user statistics** (photos, revenue, activity)
- **Credit management tools** (set, add, subtract credits)
- **User filtering and search**
- **Bulk operations** (activate/deactivate multiple users)
- **Detailed user analytics** (performance tracking)
- **Activity monitoring** (last login, last photo)

## **Implementation Priority:**
1. **Backend APIs** untuk user CRUD operations
2. **Frontend table interface** dengan filtering
3. **Credit management modals**
4. **User detail views** dengan analytics
5. **Bulk operations** dan advanced features

## **Integration Notes:**
- Connects dengan enhanced authentication system
- Uses credit system untuk accurate revenue tracking
- Integrates dengan photo generation untuk activity monitoring
- Real-time updates saat user activity changes