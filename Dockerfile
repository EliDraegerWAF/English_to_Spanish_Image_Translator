FROM python:3.11-slim
# System dependencies required by EasyOCR / OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download EasyOCR English model at build time so the first request isn't slow
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"
# Copy application code
COPY main.py image_translator.py ./
# Railway injects PORT env var; default to 8000 locally
ENV PORT=8000
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]