import uvicorn, base64, json, requests, os, gif_creator, digit_recognition, simple_ocr

from fastapi import FastAPI, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from werkzeug.utils import secure_filename
from pathlib import Path


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'}
EVENT_LOG_URL = "https://shayschwartzburd.com/EventServer/audit_log/post_event"
UPLOAD_FOLDER = Path(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "uploads"))
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],  allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )

config = {}
config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
config['CORS_HEADERS'] = 'Content-Type'


def log(app_name, app_id, message):
    try:
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
    except Exception as exception:
        print(f"Error logging to event server : {exception}")


@app.post("/PerformGifConversion")
def perform_gif_conversion(file: UploadFile):
    uploaded_file_path = upload_file(file)
    gif_bytes_io = gif_creator.create_gif(uploaded_file_path)

    log(gif_creator.APP_NAME, gif_creator.APP_ID, file.filename)
    return StreamingResponse(gif_bytes_io, media_type="image/gif",
                             headers={'Content-Disposition': f'attachment; filename="{file.filename}"'})


@app.post("/PerformOcr")
def perform_ocr(file: UploadFile):
    uploaded_file_path = upload_file(file)
    text_from_file = simple_ocr.perform_ocr_on_file(uploaded_file_path)
    
    log(simple_ocr.APP_NAME, simple_ocr.APP_ID, file.filename)
    return text_from_file


@app.post("/PerformDigitRecognition")
async def perform_digit_recognition(request: Request):
    body = await request.body()
    uploaded_file_path = upload_base64(body)
    digits_from_file = digit_recognition.perform_digit_recognition(
        uploaded_file_path)
    output_image_png_file = open(digit_recognition.DIGIT_RECOGNITION_OUTPUT_IMAGE_AS_PNG_FILE_NAME, 'rb').read()
    
    log(digit_recognition.APP_NAME, digit_recognition.APP_ID, digits_from_file)
    return base64.b64encode(output_image_png_file)


def upload_base64(base64_string):
    file_path = os.path.join(config['UPLOAD_FOLDER'],
                             digit_recognition.DIGIT_RECOGNITION_INPUT_IMAGE_AS_BASE64_FILE_NAME)
    with open(file_path, "wb") as f:
        f.write(base64_string)

    return file_path


def upload_file(file):
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb+') as f:
            file.file.seek(0)
            f.write(file.file.read())
        return file_path


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


uvicorn.run(app, host="0.0.0.0", port=3000)
