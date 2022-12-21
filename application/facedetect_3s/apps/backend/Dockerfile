FROM python:3.6.8

pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt

CMD ["python3", "backend.py"]


