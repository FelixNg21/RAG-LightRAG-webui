# syntax=docker/dockerfile:1

FROM python:3.11.9-slim

WORKDIR /flask_app
COPY requirements.txt /flask_app
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /flask_app

CMD [ "python3", "-m" , "flask","--app", "app", "run", "--host=0.0.0.0"]