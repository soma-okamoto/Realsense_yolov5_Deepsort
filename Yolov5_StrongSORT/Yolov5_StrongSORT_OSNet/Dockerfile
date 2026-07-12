# syntax=docker/dockerfile:1
# GTX 1650 / ROS Noetic / Python 3.8 / PyTorch 2.0.1 + CUDA 11.8
#
# Build context:
#   Yolov5_StrongSORT_OSNet/
#     Dockerfile
#     docker-entrypoint.sh
#     requirements.txt
#     yolov5/requirements.txt

FROM ros:noetic-ros-base-focal

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# ROS, OpenCV GUI and Python build/runtime packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash-completion \
      build-essential \
      ca-certificates \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      ros-noetic-cv-bridge \
      ros-noetic-geometry-msgs \
      ros-noetic-image-transport \
      ros-noetic-message-filters \
      ros-noetic-move-base-msgs \
      ros-noetic-sensor-msgs \
      ros-noetic-std-msgs \
      ros-noetic-tf2-geometry-msgs \
      ros-noetic-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8-compatible packaging tools.
RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==23.3.2 \
      setuptools==68.2.2 \
      wheel==0.41.3

# -----------------------------------------------------------------------------
# GPU-enabled PyTorch for CUDA 11.8.
# The cu118 wheels include the CUDA user-space runtime required by PyTorch.
# The host still needs an NVIDIA driver and NVIDIA Container Toolkit.
# -----------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
      torch==2.0.1+cu118 \
      torchvision==0.15.2+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------------------------------
# StrongSORT / YOLOv5 requirements
# Keep the CUDA-enabled torch packages installed above and constrain old-project
# dependencies to versions that still support Ubuntu 20.04 / Python 3.8.
# -----------------------------------------------------------------------------
WORKDIR /opt/strongsort-build
COPY requirements.txt /tmp/strongsort-requirements.txt
COPY yolov5/requirements.txt /tmp/yolov5-requirements.txt

RUN sed -E \
      -e '/^[[:space:]]*torch([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*torchvision([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*tb-nightly([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*tensorboard([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      /tmp/strongsort-requirements.txt > /tmp/strongsort-clean.txt \
    && sed -E \
      -e '/^[[:space:]]*torch([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*torchvision([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*tensorboard([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      /tmp/yolov5-requirements.txt > /tmp/yolov5-clean.txt \
    && printf '%s\n' \
      'Cython==0.29.36' \
      'h5py==3.10.0' \
      'imageio==2.34.2' \
      'ipython==8.12.3' \
      'matplotlib==3.7.5' \
      'numpy==1.23.5' \
      'opencv-python==4.8.1.78' \
      'pandas==1.5.3' \
      'Pillow==9.5.0' \
      'protobuf==3.20.1' \
      'psutil==5.9.8' \
      'scipy==1.10.1' \
      'seaborn==0.12.2' \
      'tensorboard==2.13.0' \
      'tqdm==4.66.5' \
      > /tmp/constraints.txt \
    && python3 -m pip install --no-cache-dir \
      -c /tmp/constraints.txt \
      -r /tmp/strongsort-clean.txt \
      -r /tmp/yolov5-clean.txt \
      tensorboard==2.13.0 \
      pyrealsense2==2.50.0.3812 \
    && python3 -m pip check

# Fail the image build if a CPU-only torch wheel was installed accidentally.
RUN python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
assert "+cu118" in torch.__version__, torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
PY

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Match the host workspace path so an existing catkin devel/setup.bash remains
# valid when ~/catkin_ws is bind-mounted from the dars account.
WORKDIR /home/dars/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
