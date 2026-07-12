#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

if [ -f /opt/ros_py310/setup.bash ]; then
  source /opt/ros_py310/setup.bash
fi

# Source a mounted catkin workspace when available.
for ws in /home/dars/catkin_ws /root/catkin_ws /catkin_ws; do
  if [ -f "${ws}/devel/setup.bash" ]; then
    source "${ws}/devel/setup.bash"
    break
  fi
done

exec "$@"
