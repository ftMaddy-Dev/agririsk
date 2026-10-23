#!/usr/bin/env bash
source .venv/bin/activate
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".vrt,.tif,.tiff"
export GDAL_HTTP_TIMEOUT=120 GDAL_HTTP_MAX_RETRY=5 GDAL_HTTP_RETRY_DELAY=5
OUTDIR=data/raw/soilgrids_tn
mkdir -p "$OUTDIR"
for PROP in phh2o sand silt clay; do
  OUT="$OUTDIR/${PROP}_0-5cm_mean.tif"
  [ -f "$OUT" ] && { echo "$PROP already done"; continue; }
  URL=$(python -c "from src.data_collection.soilgrids_client import _layer_url; print(_layer_url('$PROP','0-5cm','mean'))")
  for ATTEMPT in $(seq 1 30); do
    echo "$(date +%T) $PROP attempt $ATTEMPT"
    rm -f "$OUT.part"
    if gdal_translate -q -of GTiff -co COMPRESS=DEFLATE -co TILED=YES \
        -projwin_srs EPSG:4326 -projwin 76.0 13.8 80.6 7.9 \
        "$URL" "$OUT.part"; then
      mv "$OUT.part" "$OUT"; echo "$PROP done"; break
    fi
    sleep $((10 * ATTEMPT > 120 ? 120 : 10 * ATTEMPT))
  done
done
ls -lh "$OUTDIR"
