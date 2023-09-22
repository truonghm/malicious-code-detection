FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

COPY ./api /api
COPY ./src /api/src
COPY ./models/codebert-bimodal/checkpoint-best-aver/ /api/models


WORKDIR /api/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

ENV API_HOST_PORT=8501
ENV API_HOST_DOMAIN="0.0.0.0"
ENV RELOAD_CODE="False"
ENV NUMBER_OF_WORKERS=4

CMD [ "python", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]