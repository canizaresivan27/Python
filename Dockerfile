FROM python:3.10.12

COPY . /home/ESP32

WORKDIR /home/ESP32

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
