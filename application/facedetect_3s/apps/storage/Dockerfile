FROM python:3.6.8

WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt

CMD ["python3", "storage.py"]