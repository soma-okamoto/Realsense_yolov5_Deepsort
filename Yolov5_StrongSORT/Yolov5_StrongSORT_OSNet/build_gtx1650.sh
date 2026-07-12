#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="yolov5-strongsort:gtx1650-cu118"

docker build -t "${IMAGE_NAME}" .
printf '\nBuilt: %s\n' "${IMAGE_NAME}"
