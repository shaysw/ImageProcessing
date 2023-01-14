FROM python:3.9
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r ./requirements.txt
CMD [ "python", "flask_app.py" ]