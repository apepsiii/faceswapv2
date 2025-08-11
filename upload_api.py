import os
import time
import requests

UPLOAD_SERVER_URL_RESULT = "http://103.59.95.203:5050/upload"        # Untuk hasil biasa
UPLOAD_SERVER_URL_AR = "http://103.59.95.203:5050/upload_ar"         # Untuk hasil AR

FOLDERS_TO_MONITOR = {
    "static/results": UPLOAD_SERVER_URL_RESULT,
    "static/ar_results": UPLOAD_SERVER_URL_AR
}

already_seen = {}

for folder in FOLDERS_TO_MONITOR:
    already_seen[folder] = set(os.listdir(folder))
    print(f"[INFO] Monitoring folder: {folder}")

while True:
    for folder, target_url in FOLDERS_TO_MONITOR.items():
        current_files = set(os.listdir(folder))
        new_files = current_files - already_seen[folder]

        for file in new_files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(folder, file)
                try:
                    with open(filepath, "rb") as f:
                        response = requests.post(target_url, files={"file": (file, f)})
                    if response.status_code == 200:
                        print(f"[✓] Berhasil upload {file} ke {target_url}")
                    else:
                        print(f"[!] Gagal upload {file}, Status: {response.status_code}")
                except Exception as e:
                    print(f"[!] Error upload {file}: {e}")

        already_seen[folder] = current_files

    time.sleep(4)