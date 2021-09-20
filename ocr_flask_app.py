try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os
from pathlib import Path

from flask import Flask, request, flash, url_for
from werkzeug.utils import secure_filename, redirect

app = Flask(__name__)


UPLOAD_FOLDER = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "uploads"))
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def perform_ocr_on_file(uploaded_file_path):
    return(pytesseract.image_to_string(uploaded_file_path))


@app.route("/PerformOcr", methods=['POST'])
def perform_ocr():
    print(request)
    if 'file' not in request.files:
        flash('No file part')
        return 'No selected file'
    file = request.files['file']
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


app.run(port=3000)
