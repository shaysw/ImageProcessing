FROM python:3.9
COPY . ./home
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install tesseract-ocr -y
WORKDIR "/home"
RUN pip install -r ./requirements.txt
EXPOSE 3000
CMD [ "python", "api.py" ]