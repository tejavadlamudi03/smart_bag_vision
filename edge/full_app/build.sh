#!/usr/bin/env bash
# (earlier variant: robots + camera on the IPC - needs USB on the IPC)
# Builds the BagVision Docker image and saves it as a .tar file to load into the
# Industrial Edge App Publisher (or "docker load" on a plain IPC). Needs Docker.
#   edge/full_app/build.sh                    # CPU version (works on every IPC)
#   TORCH=cu130 edge/full_app/build.sh        # NVIDIA GPU version
#   VERSION=1.0.1 edge/full_app/build.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=${VERSION:-1.0.0}
TORCH=${TORCH:-cpu}
STAGE=edge/full_app/stage

rm -rf "$STAGE"
mkdir -p "$STAGE/app" "$STAGE/seed/bag_data" "$STAGE/weights"
cp edge/full_app/requirements.txt edge/full_app/entrypoint.sh "$STAGE/"
# program
cp ./*.py dashboard_ui.html so101_follower.urdf "$STAGE/app/"
# taught jobs, arm settings, trained models and the training crops (not the raw photos)
cp -a jobs so100_jobs so100_settings.json "$STAGE/seed/"
rsync -a --exclude run bag_models "$STAGE/seed/"
cp -a bag_data/areas.json bag_data/empty_board.png bag_data/crops "$STAGE/seed/bag_data/"
cp ~/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth "$STAGE/weights/"

docker build -f edge/full_app/Dockerfile --build-arg TORCH="$TORCH" -t "bagvision:$VERSION" "$STAGE"
docker save "bagvision:$VERSION" -o "edge/full_app/bagvision_$VERSION.tar"
echo "image saved: edge/full_app/bagvision_$VERSION.tar"
