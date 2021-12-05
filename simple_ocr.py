APP_NAME = "SimpleOCR"
APP_ID = 117

try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract




def perform_ocr_on_file(uploaded_file_path):
    return pytesseract.image_to_string(uploaded_file_path)

