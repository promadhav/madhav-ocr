FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub \
    MODEL_SOURCE=Qwen/Qwen2.5-VL-3B-Instruct \
    MODEL_REVISION=6628554 \
    OCR_SOCKET=/tmp/madhav-ocr.sock \
    OUTPUT_DIR=/app/output

COPY requirements.txt install_deps.py /app/
RUN python3 /app/install_deps.py
COPY app.py worker.py /app/
RUN mkdir -p /app/input /app/output /models/huggingface

# Default avoids a second 7.52 GB model download on the Windows build machine.
# The AMD runtime must then download weights within the 10-minute startup gate.
# Use INCLUDE_MODEL=1 only after checking disk capacity, data use and image size.
ARG INCLUDE_MODEL=0
RUN if [ "$INCLUDE_MODEL" = "1" ]; then \
      python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_SOURCE'], revision=os.environ['MODEL_REVISION'], cache_dir=os.environ['HF_HUB_CACHE'], allow_patterns=['*.json','*.safetensors','*.txt'])"; \
    elif [ "$INCLUDE_MODEL" != "0" ]; then exit 2; fi

# No exposed TCP service; the grader execs app.py repeatedly.
ENTRYPOINT []
CMD ["python3", "/app/worker.py"]
