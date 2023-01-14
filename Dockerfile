FROM python:3.9
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install tesseract-ocr -y
RUN pip install -r ./requirements.txt
EXPOSE 3000
CMD [ "python", "flask_app.py" ]