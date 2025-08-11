# 📊 CONTEXT: ADMIN DASHBOARD IMPLEMENTATION

## **Current State:**
- No admin interface exists
- Basic user dashboard at `/dashboard`
- No analytics atau reporting system
- Manual data checking via database

## **Target State:**
- Complete admin dashboard seperti design yang ditunjukkan
- Real-time analytics dengan charts
- User management tools
- Revenue tracking dan reporting
- Mobile-responsive interface

## **Dashboard Components Required:**

### **1. Top Statistics Cards:**
```javascript
// API data structure
{
    "total_users": 21,
    "total_face_swap": 1250, 
    "total_ar_photos": 890,
    "total_revenue": 250000000,
    "today_stats": {
        "face_swap": 45,
        "ar_photos": 32, 
        "revenue": 150000,
        "transactions": 30
    }
}
```

### **2. Sales Statistics (Line Chart):**
```sql
-- Daily revenue trend (30 days)
SELECT 
    DATE(settled_at) as date,
    SUM(amount) as revenue,
    COUNT(*) as transaction_count
FROM transactions 
WHERE status = 'settlement' 
    AND settled_at >= DATE('now', '-30 days')
GROUP BY DATE(settled_at)
ORDER BY date
```

### **3. Usage Statistics (Bar Chart):**
```sql
-- Daily photo usage (30 days)  
SELECT 
    DATE(created_at) as date,
    SUM(CASE WHEN photo_type = 'face_swap' THEN 1 ELSE 0 END) as face_swap,
    SUM(CASE WHEN photo_type = 'ar_photo' THEN 1 ELSE 0 END) as ar_photo
FROM photos 
WHERE created_at >= DATE('now', '-30 days')
GROUP BY DATE(created_at)
ORDER BY date
```

### **4. Photo by User (Pie Chart):**
```sql
-- User photo distribution
SELECT 
    u.username,
    COUNT(p.id) as photo_count,
    ROUND(COUNT(p.id) * 100.0 / (SELECT COUNT(*) FROM photos), 2) as percentage
FROM users u
LEFT JOIN photos p ON u.id = p.user_id
WHERE u.role = 'user'
GROUP BY u.id, u.username
ORDER BY photo_count DESC
```

## **Backend API Structure:**

### **Main Dashboard API:**
```python
@app.get("/api/admin/dashboard/stats")
async def get_dashboard_stats(admin_user = Depends(admin_required)):
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

@app.get("/api/admin/dashboard/sales-chart")
async def get_sales_chart(period: str = "30days", admin_user = Depends(admin_required)):
    # Implementation untuk sales trend data
    
@app.get("/api/admin/dashboard/usage-chart") 
async def get_usage_chart(period: str = "30days", admin_user = Depends(admin_required)):
    # Implementation untuk usage trend data
    
@app.get("/api/admin/dashboard/user-photos-pie")
async def get_user_photos_pie(admin_user = Depends(admin_required)):
    # Implementation untuk user distribution data
```

## **Frontend Structure (`pages/dashboard_admin.html`):**

### **HTML Layout:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Admin Dashboard - Platinum Cineplex</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <!-- Sidebar Navigation -->
    <div class="sidebar">
        <div class="logo">Admin Panel</div>
        <nav>
            <a href="#" class="nav-item active" data-page="dashboard">Dashboard</a>
            <a href="#" class="nav-item" data-page="users">User</a>
            <a href="#" class="nav-item" data-page="settings">Setting</a>
        </nav>
        <button class="logout-btn">Logout</button>
    </div>
    
    <!-- Main Content -->
    <div class="main-content">
        <!-- Top Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-number" id="totalUsers">-</div>
                <div class="stat-label">Total User</div>
            </div>
            <!-- More cards... -->
        </div>
        
        <!-- Charts Section -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Sale Statistic</h3>
                <canvas id="salesChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Usage Statistic</h3> 
                <canvas id="usageChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Photo by User</h3>
                <canvas id="userPieChart"></canvas>
            </div>
        </div>
    </div>
</body>
</html>
```

### **JavaScript Dashboard Logic:**
```javascript
class AdminDashboard {
    constructor() {
        this.charts = {};
        this.init();
    }
    
    async init() {
        await this.loadDashboardStats();
        this.initCharts();
        this.setupAutoRefresh();
    }
    
    async loadDashboardStats() {
        try {
            const response = await fetch('/api/admin/dashboard/stats', {
                headers: {'Authorization': `Bearer ${localStorage.getItem('token')}`}
            });
            const data = await response.json();
            
            if (data.success) {
                this.updateStatsCards(data);
            }
        } catch (error) {
            console.error('Failed to load dashboard stats:', error);
        }
    }
    
    updateStatsCards(data) {
        document.getElementById('totalUsers').textContent = data.total_users;
        document.getElementById('totalFaceSwap').textContent = data.total_face_swap;
        document.getElementById('totalARPhotos').textContent = data.total_ar_photos;
        document.getElementById('totalRevenue').textContent = this.formatCurrency(data.total_revenue);
        
        // Update today's stats
        document.getElementById('todayFaceSwap').textContent = data.today_stats.face_swap;
        document.getElementById('todayARPhotos').textContent = data.today_stats.ar_photos;
        document.getElementById('todayRevenue').textContent = this.formatCurrency(data.today_stats.revenue);
    }
    
    async initCharts() {
        await this.createSalesChart();
        await this.createUsageChart();
        await this.createUserPieChart();
    }
    
    async createSalesChart() {
        const response = await fetch('/api/admin/dashboard/sales-chart', {
            headers: {'Authorization': `Bearer ${localStorage.getItem('token')}`}
        });
        const data = await response.json();
        
        const ctx = document.getElementById('salesChart').getContext('2d');
        this.charts.sales = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.data.map(item => item.date),
                datasets: [{
                    label: 'Revenue',
                    data: data.data.map(item => item.revenue),
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return 'Rp ' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('id-ID', {
            style: 'currency',
            currency: 'IDR',
            minimumFractionDigits: 0
        }).format(amount);
    }
    
    setupAutoRefresh() {
        // Refresh dashboard every 30 seconds
        setInterval(() => {
            this.loadDashboardStats();
        }, 30000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new AdminDashboard();
});
```

## **CSS Styling (Sesuai Design):**
```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Poppins', sans-serif;
}

body {
    background: #f5f6fa;
    display: flex;
    min-height: 100vh;
}

.sidebar {
    width: 250px;
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

.sidebar .logo {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 40px;
    text-align: center;
}

.nav-item {
    display: block;
    padding: 15px 20px;
    color: rgba(255,255,255,0.8);
    text-decoration: none;
    border-radius: 10px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.nav-item.active,
.nav-item:hover {
    background: rgba(255,255,255,0.1);
    color: white;
}

.main-content {
    flex: 1;
    padding: 30px;
    overflow-y: auto;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.stat-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
    gap: 15px;
}

.stat-icon {
    font-size: 2rem;
    padding: 15px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2d3748;
}

.stat-label {
    font-size: 0.9rem;
    color: #718096;
    margin-top: 5px;
}

.charts-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 20px;
}

.chart-container {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.chart-container h3 {
    margin-bottom: 20px;
    color: #2d3748;
    font-size: 1.1rem;
}

@media (max-width: 768px) {
    body {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: auto;
    }
    
    .charts-grid {
        grid-template-columns: 1fr;
    }
}
```

## **Implementation Priority:**
1. **Create `pages/dashboard_admin.html`** dengan layout dan styling
2. **Implement backend APIs** untuk dashboard statistics
3. **Add Chart.js integration** untuk visualizations
4. **Test real-time data updates**
5. **Add mobile responsiveness**

## **Key Features:**
- Real-time dashboard updates setiap 30 detik
- Responsive design untuk mobile/tablet
- Interactive charts dengan Chart.js
- Role-based access control
- Currency formatting untuk revenue
- Period filtering untuk charts (30 days, 90 days, etc)

## **Next Integration Steps:**
1. Connect dengan enhanced authentication system
2. Integrate dengan credit system untuk accurate revenue tracking
3. Add export functionality untuk charts
4. Implement user management dari dashboard