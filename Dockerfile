FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    BBAT_WORK=/work

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages insightface
RUN pip3 uninstall -y onnxruntime || true
RUN pip3 install --no-cache-dir --break-system-packages \
        onnxruntime-gpu opencv-python-headless requests \
        faiss-cpu pandas openpyxl

RUN python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition']).prepare(ctx_id=-1)"

WORKDIR /app
COPY dedup/ ./dedup/
WORKDIR /app/dedup

ENTRYPOINT ["python3", "embed.py"]
