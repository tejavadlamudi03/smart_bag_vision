# BagVision cell (uArm + 2x SO-100 + RealSense + bag inspection) for a Siemens SIMATIC IPC /
# Industrial Edge device. Build with edge/build.sh - it prepares the build folder (edge/stage).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libusb-1.0-0 udev \
    && rm -rf /var/lib/apt/lists/*

# TORCH=cpu (any IPC) or TORCH=cu130 (IPC with an NVIDIA GPU + NVIDIA container runtime)
ARG TORCH=cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/${TORCH} \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# the IPC usually has no internet: ship the WideResNet-50 weights of the anomaly model
COPY weights/ /root/.cache/torch/hub/checkpoints/

WORKDIR /app
COPY app/ /app/
COPY seed/ /app/seed/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1 YOLO_OFFLINE=1 PORT=8000
EXPOSE 8000
VOLUME ["/data"]
ENTRYPOINT ["/entrypoint.sh"]
