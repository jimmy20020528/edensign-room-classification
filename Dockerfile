# CPU-only room-classification endpoint. No GPU required.
FROM python:3.11-slim

WORKDIR /srv

# System libs needed by Pillow / torch image loading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller than the default CUDA build),
# then the rest of the deps explicitly (so torch is NOT pulled from PyPI/CUDA).
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        fastapi==0.115.0 uvicorn==0.30.0 python-multipart==0.0.12 \
        numpy==1.26.4 transformers==4.44.2 scikit-learn==1.5.2 \
        joblib==1.4.2 Pillow==10.4.0

# Pre-download DINOv2-base so the container is ready immediately (no first-call download)
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
AutoModel.from_pretrained('facebook/dinov2-base'); print('DINOv2 cached')"

COPY app/ app/
COPY scripts/ scripts/
COPY artifacts/ artifacts/

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
