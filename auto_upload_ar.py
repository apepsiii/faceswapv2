import os
import time
import requests

UPLOAD_FOLDER = "static/ar_results"
UPLOAD_SERVER_URL = "http://103.59.95.203:5050/upload_ar"  # Endpoint baru buat foto AR di server

def monitor_and_upload_ar():
    print(f"[INFO] Monitoring folder AR: {UPLOAD_FOLDER}")
    already_seen = set(os.listdir(UPLOAD_FOLDER))

    while True:
        current_files = set(os.listdir(UPLOAD_FOLDER))
        new_files = current_files - already_seen

        for file in new_files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(UPLOAD_FOLDER, file)
                try:
                    time.sleep(4)  # Kasih delay 4 detik biar yakin file udah siap
                    with open(filepath, "rb") as f:
                        response = requests.post(UPLOAD_SERVER_URL, files={"file": (file, f)})
                    print(f"[✓] Upload AR {file} => {response.status_code}")
                except Exception as e:
                    print(f"[!] Gagal upload AR {file}: {e}")

        already_seen = current_files
        time.sleep(4)

if __name__ == "__main__":
    monitor_and_upload_ar()