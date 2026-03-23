FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install -y --no-install-recommends openjdk-11-jre-headless build-essential ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY src/ /app/src/

# Copy models if present (best-effort; build will not fail if models/ absent)
COPY models/ /app/models/  

ENTRYPOINT ["python", "src/predict.py"]
