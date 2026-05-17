# ===== BUILD STAGE =====
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install  --prefix=/install -r requirements.txt


# ===== RUNTIME STAGE =====
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier uniquement les dépendances installées depuis le builder
COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]