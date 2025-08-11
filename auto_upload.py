
import os
import time
import requests

UPLOAD_SERVER_URL = "http://103.59.95.203:5050/upload"  # Ganti sesuai IP server kamu
RESULTS_FOLDER = "static/results"

def monitor_and_upload():
    print(f"[INFO] Monitoring folder: {RESULTS_FOLDER}")
    already_seen = set(os.listdir(RESULTS_FOLDER))

    while True:
        current_files = set(os.listdir(RESULTS_FOLDER))
        new_files = current_files - already_seen

        for file in new_files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(RESULTS_FOLDER, file)
                try:
                    with open(filepath, "rb") as f:
                        response = requests.post(UPLOAD_SERVER_URL, files={"file": (file, f)})
                    print(f"[✓] Upload {file} => {response.status_code}")
                except Exception as e:
                    print(f"[!] Gagal upload {file}: {e}")

        already_seen = current_files
        time.sleep(2)

if __name__ == "__main__":
    monitor_and_upload()
