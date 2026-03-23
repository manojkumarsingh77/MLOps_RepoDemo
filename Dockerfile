FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \ 
       build-essential \ 
       ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/

RUN mkdir -p /app/models

ENTRYPOINT ["python", "src/predict.py"]
