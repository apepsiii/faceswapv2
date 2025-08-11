from flask import Flask, request
import os

app = Flask(__name__)

RESULTS_FOLDER = '/var/www/faceswap/backend/static/results'
AR_RESULTS_FOLDER = '/var/www/faceswap/backend/static/ar_results'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    save_path = os.path.join(RESULTS_FOLDER, file.filename)
    file.save(save_path)
    print(f"File {file.filename} berhasil diterima dan disimpan di {save_path}")

    return 'File uploaded successfully', 200

@app.route('/upload_ar', methods=['POST'])
def upload_file_ar():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    save_path = os.path.join(AR_RESULTS_FOLDER, file.filename)
    file.save(save_path)
    print(f"File AR {file.filename} berhasil diterima dan disimpan di {save_path}")

    return 'File AR uploaded successfully', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)