#!/bin/sh
# Everything the cell learns or produces (jobs, models, photos, settings) lives on the /data volume,
# so it survives app updates. On the first start /data is filled from the copy shipped in the image.
set -e
for item in jobs so100_jobs so100_settings.json so100_logs bag_models bag_data captures inspections; do
    if [ ! -e "/data/$item" ]; then
        if [ -e "/app/seed/$item" ]; then
            cp -a "/app/seed/$item" "/data/$item"
        else
            mkdir -p "/data/$item"
        fi
    fi
    ln -sfn "/data/$item" "/app/$item"
done
exec python3 -u /app/dashboard.py --port "${PORT:-8000}"
