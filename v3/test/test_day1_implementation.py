import requests
import json
import sqlite3
from datetime import datetime, timedelta
import os

class Day1Tester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.admin_token = None
        self.db_path = "face_swap.db"
        
    def setup_test_data(self):
        """Setup test data in database for testing"""
        print("🔧 Setting up test data...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # *** PERBAIKAN: Insert real transaction data sesuai sistem yang ada ***
                test_transactions = [
                    (1, "ORDER-test001", 5000, 3, "settlement", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    (1, "ORDER-test002", 5000, 3, "settlement", datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 
                    (2, "ORDER-test003", 5000, 3, "settlement", (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')),
                    (3, "ORDER-test004", 5000, 3, "settlement", (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')),
                    (1, "ORDER-test005", 5000, 3, "pending", datetime.now().strftime('%Y-%m-%d %H:%M:%S')),  # Pending - tidak dihitung
                ]
                
                for user_id, order_id, amount, credits, status, settled_at in test_transactions:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO transactions 
                            (user_id, order_id, amount, credits_added, status, settled_at, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (user_id, order_id, amount, credits, status, settled_at, settled_at))
                    except sqlite3.OperationalError as e:
                        print(f"⚠️  Error inserting transaction {order_id}: {e}")
                        
                # Insert test face swap history for last 7 days
                for i in range(7):
                    date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S')
                    count = 5 - i  # Decreasing count for visual effect
                    
                    for j in range(count):
                        try:
                            conn.execute("""
                                INSERT INTO face_swap_history 
                                (user_id, template_name, result_filename, created_at)
                                VALUES (?, ?, ?, ?)
                            """, (1, 'test_template.jpg', f'test_result_{i}_{j}.jpg', date))
                        except sqlite3.OperationalError as e:
                            print(f"⚠️  Error inserting face swap history: {e}")
                
                conn.commit()
                print("✅ Test data setup completed")
                
                # *** VALIDASI: Check actual revenue dari test data ***
                cursor = conn.execute("SELECT SUM(amount) FROM transactions WHERE status = 'settlement'")
                test_revenue = cursor.fetchone()[0] or 0
                print(f"💰 Test revenue setup: Rp {test_revenue:,}")
                
        except Exception as e:
            print(f"❌ Error setting up test data: {e}")
    
    def test_admin_login(self):
        """Test admin login and get token"""
        print("\n🔑 Testing admin login...")
        
        try:
            response = requests.post(f"{self.base_url}/api/login", json={
                "username": "admin",
                "password": "admin123"
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.admin_token = data.get("token")
                    print(f"✅ Admin login successful")
                    print(f"   Token: {self.admin_token[:20]}...")
                    return True
                else:
                    print(f"❌ Admin login failed: {data.get('message')}")
            else:
                print(f"❌ Admin login failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Admin login error: {e}")
            
        return False
    
    def test_revenue_calculation(self):
        """Test real revenue calculation from transactions table"""
        print("\n💰 Testing REAL revenue calculation...")
        
        if not self.admin_token:
            print("❌ No admin token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.admin_token}"}
            response = requests.get(f"{self.base_url}/api/admin/dashboard/stats", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    total_revenue = data.get("total_revenue", 0)
                    today_revenue = data.get('today_stats', {}).get('revenue', 0)
                    
                    print(f"✅ Revenue calculation working")
                    print(f"   Total Revenue: Rp {total_revenue:,}")
                    print(f"   Today Revenue: Rp {today_revenue:,}")
                    print(f"   Total Users: {data.get('total_users', 0)}")
                    print(f"   Total Face Swap: {data.get('total_face_swap', 0)}")
                    print(f"   Total AR Photos: {data.get('total_ar_photos', 0)}")
                    print(f"   Today Transactions: {data.get('today_stats', {}).get('transactions', 0)}")
                    
                    # *** VALIDASI: Check revenue calculation method ***
                    # Validate dengan direct database query
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute("""
                            SELECT COALESCE(SUM(amount), 0) 
                            FROM transactions 
                            WHERE status = 'settlement'
                        """)
                        db_revenue = cursor.fetchone()[0]
                        
                        if total_revenue == db_revenue:
                            print(f"✅ Revenue calculation CORRECT: API matches database")
                            print(f"   Database total: Rp {db_revenue:,}")
                        else:
                            print(f"❌ Revenue calculation MISMATCH!")
                            print(f"   API total: Rp {total_revenue:,}")
                            print(f"   Database total: Rp {db_revenue:,}")
                            return False
                    
                    # Check that revenue is from transactions, not photo count
                    if total_revenue > 0:
                        print("✅ Revenue using REAL transaction amounts (not photo count)")
                    else:
                        print("⚠️  Revenue is 0, check if test transactions exist")
                        
                    return True
                else:
                    print(f"❌ API returned error: {data}")
            else:
                print(f"❌ API request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Revenue calculation test error: {e}")
            
        return False
    
    def test_photo_activity_chart(self):
        """Test photo activity chart for 7 days"""
        print("\n📊 Testing photo activity chart...")
        
        if not self.admin_token:
            print("❌ No admin token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.admin_token}"}
            response = requests.get(f"{self.base_url}/api/admin/dashboard/photo-activity-7days", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    labels = data.get("labels", [])
                    activity_data = data.get("data", [])
                    
                    print(f"✅ Photo activity chart working")
                    print(f"   Period: {data.get('period', 'unknown')}")
                    print(f"   Days: {', '.join(labels)}")
                    print(f"   Data points: {len(activity_data)}")
                    
                    # Show sample data
                    for item in activity_data[-3:]:  # Show last 3 days
                        print(f"   {item['day']}: Face Swap={item['face_swap']}, AR={item['ar_photos']}")
                        
                    return True
                else:
                    print(f"❌ API returned error: {data}")
            else:
                print(f"❌ API request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Photo activity chart test error: {e}")
            
        return False
    
    def test_dashboard_page(self):
        """Test dashboard page accessibility"""
        print("\n🖥️  Testing dashboard page...")
        
        try:
            response = requests.get(f"{self.base_url}/dashboard_admin")
            
            if response.status_code == 200:
                content = response.text
                
                # Check for key elements
                checks = [
                    ("Photo Activity Chart", "photoActivityChart" in content),
                    ("Chart.js Library", "chart.js" in content),
                    ("Dashboard Stats", "totalRevenue" in content),
                    ("Admin Dashboard Class", "AdminDashboard" in content),
                    ("Photo Activity Function", "loadPhotoActivityChart" in content)
                ]
                
                print("✅ Dashboard page accessible")
                for check_name, result in checks:
                    status = "✅" if result else "❌"
                    print(f"   {status} {check_name}")
                    
                return all(result for _, result in checks)
            else:
                print(f"❌ Dashboard page not accessible (status {response.status_code})")
                
        except Exception as e:
            print(f"❌ Dashboard page test error: {e}")
            
        return False
    
    def test_api_endpoints(self):
        """Test all new API endpoints"""
        print("\n🔌 Testing API endpoints...")
        
        if not self.admin_token:
            print("❌ No admin token available")
            return False
            
        endpoints = [
            "/api/admin/dashboard/stats",
            "/api/admin/dashboard/photo-activity-7days",
            "/api/admin/dashboard/activity-chart"
        ]
        
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        results = []
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    success = data.get("success", False)
                    print(f"   ✅ {endpoint} - Status: {response.status_code}, Success: {success}")
                    results.append(success)
                else:
                    print(f"   ❌ {endpoint} - Status: {response.status_code}")
                    results.append(False)
                    
            except Exception as e:
                print(f"   ❌ {endpoint} - Error: {e}")
                results.append(False)
        
        return all(results)
    
    def test_database_queries(self):
        """Test database queries for revenue and photo activity"""
        print("\n🗄️  Testing database queries...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # *** TEST 1: Revenue query (MOST IMPORTANT) ***
                print("   📊 Testing REVENUE calculation...")
                cursor = conn.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM transactions 
                    WHERE status = 'settlement'
                """)
                total_revenue = cursor.fetchone()[0]
                print(f"   ✅ Total Revenue query: Rp {total_revenue:,}")
                
                # Count settlement transactions
                cursor = conn.execute("SELECT COUNT(*) FROM transactions WHERE status = 'settlement'")
                settlement_count = cursor.fetchone()[0]
                print(f"   ✅ Settlement transactions: {settlement_count}")
                
                # Today's revenue
                today = datetime.now().strftime('%Y-%m-%d')
                cursor = conn.execute("""
                    SELECT COALESCE(SUM(amount), 0), COUNT(*) 
                    FROM transactions 
                    WHERE status = 'settlement' AND DATE(settled_at) = ?
                """, (today,))
                today_result = cursor.fetchone()
                today_revenue = today_result[0]
                today_tx_count = today_result[1]
                print(f"   ✅ Today's revenue: Rp {today_revenue:,} ({today_tx_count} transactions)")
                
                # *** TEST 2: Photo activity query ***
                print("   📸 Testing PHOTO ACTIVITY queries...")
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM face_swap_history 
                    WHERE DATE(created_at) >= DATE('now', '-7 days')
                """)
                photos_7days = cursor.fetchone()[0]
                print(f"   ✅ Photos last 7 days: {photos_7days}")
                
                # Today's photos
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM face_swap_history 
                    WHERE DATE(created_at) = ?
                """, (today,))
                today_photos = cursor.fetchone()[0]
                print(f"   ✅ Today's photos: {today_photos}")
                
                # *** TEST 3: User revenue breakdown ***
                print("   🏪 Testing REVENUE PER SITE queries...")
                cursor = conn.execute("""
                    SELECT 
                        u.username,
                        COALESCE(SUM(t.amount), 0) as revenue
                    FROM users u
                    LEFT JOIN transactions t ON u.id = t.user_id AND t.status = 'settlement'
                    WHERE u.role = 'user'
                    GROUP BY u.username
                    ORDER BY revenue DESC
                """)
                
                site_revenues = cursor.fetchall()
                for username, revenue in site_revenues[:5]:  # Show top 5
                    print(f"   ✅ Site {username.upper()}: Rp {revenue:,}")
                
                return True
                
        except Exception as e:
            print(f"   ❌ Database query error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all Day 1 tests"""
        print("🚀 DAY 1 IMPLEMENTATION TESTING")
        print("=" * 50)
        
        # Setup test data
        self.setup_test_data()
        
        # Run tests
        tests = [
            ("Admin Login", self.test_admin_login),
            ("Revenue Calculation", self.test_revenue_calculation),
            ("Photo Activity Chart", self.test_photo_activity_chart),
            ("Dashboard Page", self.test_dashboard_page),
            ("API Endpoints", self.test_api_endpoints),
            ("Database Queries", self.test_database_queries)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} test failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n📋 TEST SUMMARY")
        print("=" * 30)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All Day 1 tests passed! Ready for Day 2.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
            
        return passed == total

def main():
    """Main testing function"""
    print("🧪 DAY 1 TESTING SUITE")
    print("Testing: Revenue Calculation + Photo Activity Chart")
    print("")
    
    # Check if server is running
    tester = Day1Tester()
    
    try:
        response = requests.get(f"{tester.base_url}/health")
        if response.status_code != 200:
            print("❌ Server not running. Please start the FastAPI server first.")
            return
    except:
        print("❌ Cannot connect to server. Please start the FastAPI server first.")
        return
    
    # Run tests
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Day 1 implementation is ready!")
        print("Next steps for Day 2:")
        print("  1. Update User Activity → Revenue Per Site")
        print("  2. Hide Credits in User Management")  
        print("  3. Create Site Users (18 locations)")
    else:
        print("\n🔧 Please fix the failing tests before proceeding to Day 2.")

if __name__ == "__main__":
    main()