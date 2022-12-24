FROM python:3.6.8

WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

CMD ["python3", "storage.py"]