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

exec "$@"
