#!/usr/bin/env bash
set -eo pipefail

# Run as root inside the RTX 5070 container, or during docker build.
CONDA_ENV="${CONDA_ENV:-/opt/conda/envs/rosgpu}"
BOOST_PREFIX="/opt/boost-python310"
ROS_DISTRO="${ROS_DISTRO:-noetic}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root inside the RTX 5070 container." >&2
  exit 1
fi

test -x "${CONDA_ENV}/bin/python3"
test -r "${BOOST_PREFIX}/lib/libboost_python310.so"
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${BOOST_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

BUILD_WS="$(mktemp -d /tmp/cv-bridge-py310.XXXXXX)"
trap 'rm -rf "${BUILD_WS}"' EXIT
mkdir -p "${BUILD_WS}/src"
git clone --depth 1 --branch noetic \
  https://github.com/ros-perception/vision_opencv.git \
  "${BUILD_WS}/src/vision_opencv"

# Noetic's generic "python" component can select Ubuntu's Python 3.8 library.
# Require the versioned component and use FindBoost instead of BoostConfig.cmake.
"${CONDA_ENV}/bin/python3" - "${BUILD_WS}/src/vision_opencv/cv_bridge/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = "find_package(Boost REQUIRED python)"
assert source.count(old) == 1, "Unexpected cv_bridge Boost configuration"
path.write_text(source.replace(old, "find_package(Boost REQUIRED python310)"))
PY

source "/opt/ros/${ROS_DISTRO}/setup.bash"
catkin config \
  --workspace "${BUILD_WS}" \
  --extend "/opt/ros/${ROS_DISTRO}" \
  --install \
  --install-space /opt/ros_py310
catkin build \
  --workspace "${BUILD_WS}" \
  cv_bridge \
  --jobs 2 \
  --parallel-packages 1 \
  --no-status \
  --summarize \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_RPATH="${BOOST_PREFIX}/lib;${CONDA_ENV}/lib" \
    -DSETUPTOOLS_DEB_LAYOUT=OFF \
    -DPYTHON_EXECUTABLE="${CONDA_ENV}/bin/python3" \
    -DPYTHON_INCLUDE_DIR="${CONDA_ENV}/include/python3.10" \
    -DPYTHON_LIBRARY="${CONDA_ENV}/lib/libpython3.10.so" \
    -DBOOST_ROOT="${BOOST_PREFIX}" \
    -DBOOST_LIBRARYDIR="${BOOST_PREFIX}/lib" \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DBoost_INCLUDE_DIR="${BOOST_PREFIX}/include" \
    -DBoost_PYTHON310_LIBRARY_RELEASE="${BOOST_PREFIX}/lib/libboost_python310.so" \
    -DBoost_PYTHON310_LIBRARY_DEBUG="${BOOST_PREFIX}/lib/libboost_python310.so"

mapfile -t BRIDGE_MODULES < <(find /opt/ros_py310/lib -name 'cv_bridge_boost*.so' -type f)
test "${#BRIDGE_MODULES[@]}" -gt 0
for module in "${BRIDGE_MODULES[@]}"; do
  LINK_INFO="$(ldd "${module}")"
  printf '%s\n' "${module}"
  grep -E 'boost_python|libpython|libcv_bridge|not found' <<< "${LINK_INFO}"
  if grep -Eq 'libboost_python38|libpython3\.8|not found' <<< "${LINK_INFO}"; then
    echo "cv_bridge still has incompatible or missing libraries." >&2
    exit 1
  fi
  grep -F 'libboost_python310.so' <<< "${LINK_INFO}" \
    | grep -F "=> ${BOOST_PREFIX}/lib/"
done

source /opt/ros_py310/setup.bash
export PYTHONPATH="/opt/ros_py310/lib/python3.10/site-packages:/opt/ros_py310/lib/python3.10/dist-packages:/opt/ros_py310/lib/python3/dist-packages:${PYTHONPATH:-}"
export PYTHONPATH="$(printf '%s' "${PYTHONPATH}" | tr ':' '\n' \
  | awk 'NF && $0 != "/usr/lib/python3/dist-packages" && !seen[$0]++' | paste -sd: -)"
SYSTEM_LIBFFI="$(ldconfig -p \
  | awk '/libffi\.so\.7 .*x86-64/ && !found {print $NF; found=1}')"
test -n "${SYSTEM_LIBFFI}"
export LD_PRELOAD="${SYSTEM_LIBFFI}${LD_PRELOAD:+:${LD_PRELOAD}}"
PYTHONNOUSERSITE=1 "${CONDA_ENV}/bin/python3" - <<'PY'
import sys
import numpy as np
import cv_bridge
from cv_bridge.boost import cv_bridge_boost

assert sys.version_info[:2] == (3, 10), sys.version
assert np.__version__ == "1.23.5", np.__version__
assert cv_bridge.__file__.startswith("/opt/ros_py310/"), cv_bridge.__file__
assert cv_bridge_boost.__file__.startswith("/opt/ros_py310/"), cv_bridge_boost.__file__
bridge = cv_bridge.CvBridge()
bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
msg = bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg, "rgb8"), bgr[:, :, ::-1])
depth = np.array([[1.25]], dtype=np.float32)
msg = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg, "passthrough"), depth)
print("cv_bridge Python 3.10 / Boost.Python 3.10 color/depth conversion OK")
PY
