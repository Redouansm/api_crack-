# syntax=docker/dockerfile:1

# ===== BUILD STAGE =====
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Cache pip conservé entre les builds : torch & co ne sont téléchargés qu'une fois
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt


# ===== RUNTIME STAGE =====
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HF_HUB_OFFLINE=1 \
    HUMIDITY_SAM_WEIGHTS=/opt/weights/FastSAM-s.pt

COPY --from=builder /install /usr/local

# FastSAM téléchargé UNE fois, avant le code (étape gardée en cache)
RUN mkdir -p /opt/weights && python -c "import urllib.request as u; u.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt', '/opt/weights/FastSAM-s.pt')" \
    || echo "FastSAM-s.pt non téléchargé (sera fait au runtime)"

WORKDIR /app

# Le code en dernier : le modifier ne refait que cette étape (quelques secondes)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]