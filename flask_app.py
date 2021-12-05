import json
import requests
from flask import Flask, request, flash
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from simple_ocr import perform_ocr_on_file

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
EVENT_LOG_URL = "http://127.0.0.1:8000/EventServer/audit_log/post_event"
APP_NAME = "SimpleOCR"
APP_ID = 117
UPLOAD_FOLDER = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "uploads"))

app = Flask(__name__)
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def log(message):
    payload = {
        "event_type": "LOG",
        "application_id": APP_ID,
        "sender": APP_NAME,
        "value": message
    }
    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.request("POST", EVENT_LOG_URL, headers=headers, data=json.dumps(payload))
    print(response.text)


@app.route("/PerformOcr", methods=['POST'])
def perform_ocr():
    if 'file' not in request.files:
        log('No file part')
        return 'No selected file'
    file = request.files['file']
    log(file.filename)
    uploaded_file_path = upload_file(file)
    text_fom_file = perform_ocr_on_file(uploaded_file_path)
    return text_fom_file


def upload_file(file):
    if file.filename == '':
        flash('No selected file')
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app.run(port=3000)
