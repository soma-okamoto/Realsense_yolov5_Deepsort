#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

if [ -f /opt/ros_py310/setup.bash ]; then
  source /opt/ros_py310/setup.bash
fi

HOST_WS="/home/dars/catkin_ws"

if [ -f "${HOST_WS}/devel/setup.bash" ]; then
  source "${HOST_WS}/devel/setup.bash"
elif [ -d "${HOST_WS}/src" ]; then
  export ROS_PACKAGE_PATH="${HOST_WS}/src:${ROS_PACKAGE_PATH:-}"
fi

# Ubuntu側のlibffi 7を優先して、
# cv_bridgeとlibp11-kitの競合を回避する
SYSTEM_LIBFFI="$(ldconfig -p \
  | awk '/libffi\.so\.7 .*x86-64/ {print $NF; exit}')"

if [ -n "${SYSTEM_LIBFFI}" ]; then
  export LD_PRELOAD="${SYSTEM_LIBFFI}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "[entrypoint] Using system libffi: ${SYSTEM_LIBFFI}"
else
  echo "[entrypoint] Warning: system libffi.so.7 was not found" >&2
fi

exec "$@"