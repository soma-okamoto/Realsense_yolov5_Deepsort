#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-yolov5-strongsort:rtx5070-cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-yolov5_strongsort_rtx5070}"

# sudo実行時も、実際のホストユーザー情報を取得する
if [[ -n "${SUDO_USER:-}" ]]; then
  HOST_USER="${SUDO_USER}"
  USER_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
  HOST_UID="${SUDO_UID}"
  HOST_GID="${SUDO_GID}"
else
  HOST_USER="$(id -un)"
  USER_HOME="${HOME}"
  HOST_UID="$(id -u)"
  HOST_GID="$(id -g)"
fi

HOST_WS="${HOST_WS:-${USER_HOME}/catkin_ws}"

ROS_MASTER_URI_VALUE="${ROS_MASTER_URI:-http://192.168.1.23:11311}"
ROS_IP_VALUE="${ROS_IP:-192.168.1.23}"

DOCKER_HOME="${DOCKER_HOME:-${USER_HOME}/.docker_yolov5_rtx5070}"

if [[ ! -d "${HOST_WS}" ]]; then
  echo "catkin workspace not found: ${HOST_WS}" >&2
  exit 1
fi

mkdir -p \
  "${DOCKER_HOME}/.ros/log" \
  "${DOCKER_HOME}/.cache" \
  "${DOCKER_HOME}/.config" \
  "${DOCKER_HOME}/.matplotlib"

if [[ "${EUID}" -eq 0 ]]; then
  chown -R "${HOST_UID}:${HOST_GID}" "${DOCKER_HOME}"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DISPLAY_ARGS=()

if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  DISPLAY_ARGS+=(
    -e "DISPLAY=${DISPLAY}"
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
fi

docker run -it \
  --name "${CONTAINER_NAME}" \
  --hostname "$(hostname)" \
  --user "${HOST_UID}:${HOST_GID}" \
  --workdir /home/dars/catkin_ws \
  --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=4g \
  --privileged \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "${DOCKER_HOME}:/home/dars" \
  -v "${HOST_WS}:/home/dars/catkin_ws" \
  -v /dev/bus/usb:/dev/bus/usb \
  -e HOME=/home/dars \
  -e USER="${HOST_USER}" \
  -e LOGNAME="${HOST_USER}" \
  -e ROS_HOME=/home/dars/.ros \
  -e ROS_LOG_DIR=/home/dars/.ros/log \
  -e XDG_CACHE_HOME=/home/dars/.cache \
  -e MPLCONFIGDIR=/home/dars/.matplotlib \
  -e "ROS_MASTER_URI=${ROS_MASTER_URI_VALUE}" \
  -e "ROS_IP=${ROS_IP_VALUE}" \
  "${DISPLAY_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash
