#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-yolov5-strongsort:rtx5070-cu128}"

docker build \
  --network=host \
  -f Dockerfile.rtx5070 \
  -t "${IMAGE_NAME}" \
  .

printf '\nBuilt: %s\n' "${IMAGE_NAME}"
