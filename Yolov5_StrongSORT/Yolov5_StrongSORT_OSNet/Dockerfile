# GTX 1650 / ROS Noetic / Python 3.8
# PyTorch 2.0.1 + CUDA 11.8

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
# ROS / OpenCV / Python packages
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
      python3-catkin-tools \
      ros-noetic-cv-bridge \
      ros-noetic-geometry-msgs \
      ros-noetic-image-transport \
      ros-noetic-message-filters \
      ros-noetic-message-generation \
      ros-noetic-message-runtime \
      ros-noetic-move-base-msgs \
      ros-noetic-sensor-msgs \
      ros-noetic-std-msgs \
      ros-noetic-tf2-geometry-msgs \
      ros-noetic-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8 compatible versions
RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==23.3.2 \
      setuptools==68.2.2 \
      wheel==0.41.3

# -----------------------------------------------------------------------------
# CUDA 11.8版 PyTorch
# -----------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
      torch==2.0.1+cu118 \
      torchvision==0.15.2+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------------------------------
# StrongSORT / YOLOv5 requirements
# -----------------------------------------------------------------------------
WORKDIR /opt/strongsort-build

COPY requirements.txt /tmp/strongsort-requirements.txt
COPY yolov5/requirements.txt /tmp/yolov5-requirements.txt

RUN sed -E \
      -e '/^[[:space:]]*torch([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*torchvision([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*tb-nightly([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*tensorboard([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*PyYAML([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      /tmp/strongsort-requirements.txt \
      > /tmp/strongsort-clean.txt \
    && sed -E \
      -e '/^[[:space:]]*torch([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*torchvision([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*tensorboard([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      -e '/^[[:space:]]*PyYAML([<>=!~ ].*)?([[:space:]]*#.*)?$/d' \
      /tmp/yolov5-requirements.txt \
      > /tmp/yolov5-clean.txt \
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
      pyrealsense2==2.50.0.3812

# -----------------------------------------------------------------------------
# StrongSORTが要求する公式torchreid
#
# PyPIのtorchreid==0.2.5では、
# torchreid.metrics.distance が存在しないため使用しない。
# -----------------------------------------------------------------------------
RUN python3 -m pip uninstall -y torchreid || true \
    && python3 -m pip install --no-cache-dir --no-deps \
      "git+https://github.com/KaiyangZhou/deep-person-reid.git@f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"


# torchreidが実行時に使用する追加依存
RUN python3 -m pip install --no-cache-dir \
      chardet==5.2.0

# 推論に必要な主要モジュールを個別確認
# torchreidのtb-nightly要求は学習環境向けなので、推論用イメージでは無視する
RUN python3 -c "import torch; \
import cv2; \
import numpy; \
import pyrealsense2; \
from torchreid.metrics.distance import compute_distance_matrix; \
print('Runtime dependency check OK')"

# PyTorch CUDA版確認
# heredocではなく、旧Docker builderでも動く形にする
RUN python3 -c "import torch; \
print('torch:', torch.__version__); \
print('torch CUDA runtime:', torch.version.cuda); \
assert '+cu118' in torch.__version__, torch.__version__; \
assert torch.version.cuda == '11.8', torch.version.cuda"

# torchreid構成確認
RUN python3 -c "from torchreid.metrics.distance import compute_distance_matrix; \
print('official torchreid OK')"

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /home/dars/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]