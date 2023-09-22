# FROM python:3.9
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /api/

COPY requirements.txt ./requirements.txt
RUN pip install --user --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "python", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]