import json
import requests
from flask import Flask, request, flash
from werkzeug.utils import secure_filename
import os
from pathlib import Path

import digit_recognition
import simple_ocr

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
EVENT_LOG_URL = "http://127.0.0.1:8000/EventServer/audit_log/post_event"
UPLOAD_FOLDER = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "uploads"))
DIGIT_RECOGNITION_IMAGE_AS_BASE64_FILE_NAME = "digit_recognition_image.png"
app = Flask(__name__)
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'


def log(app_name, app_id, message):
    payload = {
        "event_type": "LOG",
        "application_id": app_id,
        "sender": app_name,
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
    log(simple_ocr.APP_NAME, simple_ocr.APP_ID, file.filename)
    uploaded_file_path = upload_file(file)
    text_from_file = simple_ocr.perform_ocr_on_file(uploaded_file_path)
    return text_from_file


@app.route("/PerformDigitRecognition", methods=['POST'])
def perform_digit_recognition():
    uploaded_file_path = upload_base64(request.data)
    digits_from_file = digit_recognition.perform_digit_recognition(uploaded_file_path)
    log(digit_recognition.APP_NAME, digit_recognition.APP_ID, uploaded_file_path)
    return digits_from_file


def upload_base64(base64_string):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], DIGIT_RECOGNITION_IMAGE_AS_BASE64_FILE_NAME)
    with open(file_path, "wb") as f:
        f.write(base64_string)

    return file_path


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
