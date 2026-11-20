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

# ホスト側のcatkin環境を読み込んだ後でPython 3.10向けcv_bridgeを優先する。
PY310_PATHS=""
for python_dir in /opt/ros_py310/lib/python{3.10,3}/{site,dist}-packages; do
  if [ -d "${python_dir}" ]; then
    PY310_PATHS="${PY310_PATHS:+${PY310_PATHS}:}${python_dir}"
  fi
done
export PYTHONPATH="$(printf '%s' "${PY310_PATHS}:${PYTHONPATH:-}" \
  | tr ':' '\n' \
  | awk 'NF && $0 != "/usr/lib/python3/dist-packages" && !seen[$0]++' \
  | paste -sd: -)"

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
