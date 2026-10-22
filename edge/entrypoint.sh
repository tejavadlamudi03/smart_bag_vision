#!/bin/sh
# Models live on the /data volume (the dashboard updates them over the network).
# On the first start the models shipped in the image are copied there.
set -e
if [ ! -d "$BAG_MODELS" ]; then
    mkdir -p "$(dirname "$BAG_MODELS")"
    cp -a /app/models_seed "$BAG_MODELS"
fi
exec python3 -u /app/inspect_server.py --port "${PORT:-8100}"
