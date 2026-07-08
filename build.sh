#!/usr/bin/env bash
# Build + tag the NVFP4-KV image. Push manually after review:
#   docker push danielwoz/vllm:dspark-nvfp4-cu132-YYYYMMDD && docker push danielwoz/vllm:dspark-nvfp4-cu132
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
IMG=danielwoz/vllm
docker build -t "$IMG:dspark-nvfp4-cu132-$DATE_TAG" -t "$IMG:dspark-nvfp4-cu132" "$HERE"
echo ">> Built $IMG:dspark-nvfp4-cu132-$DATE_TAG"
