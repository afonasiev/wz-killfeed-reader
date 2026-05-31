FROM debian:trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-opencv python3-numpy python3-yaml tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus tesseract-ocr-jpn tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
    && rm -rf /var/lib/apt/lists/*

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python3", "-m", "warzone_analyzer"]
