#!/usr/bin/env bash
# Builds the BagVision inspection-service image and saves it as a .tar file to load into the
# Industrial Edge App Publisher (or "docker load" on a plain IPC). Needs Docker.
#   edge/build.sh                      # CPU version (works on every IPC)
#   TORCH=cu130 edge/build.sh          # NVIDIA GPU version
#   VERSION=1.0.1 edge/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=${VERSION:-1.0.0}
TORCH=${TORCH:-cpu}
STAGE=edge/stage

rm -rf "$STAGE"
mkdir -p "$STAGE/app" "$STAGE/models" "$STAGE/weights"
cp edge/requirements.txt edge/entrypoint.sh "$STAGE/"
cp inspect_server.py anomaly.py bag_inspect.py "$STAGE/app/"
# the models in use now (the dashboard sends newer ones itself)
cp bag_models/best.pt bag_models/anomaly.pt "$STAGE/models/"
[ -d bag_models/profiles ] && cp -a bag_models/profiles "$STAGE/models/"
cp ~/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth "$STAGE/weights/"

docker build -f edge/Dockerfile --build-arg TORCH="$TORCH" -t "bagvision-ai:$VERSION" "$STAGE"
docker save "bagvision-ai:$VERSION" -o "edge/bagvision-ai_$VERSION.tar"
echo "image saved: edge/bagvision-ai_$VERSION.tar"
