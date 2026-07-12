# RTX 5070版 Docker 手順

## 使用構成

- GPU: NVIDIA GeForce RTX 5070
- OS: Ubuntu 20.04 base
- ROS: Noetic
- Python: 3.10
- PyTorch: 2.7.1
- torchvision: 0.22.1
- CUDA runtime: 12.8
- RealSense Python: pyrealsense2 2.55.1.6486

PyTorch 2.7系のCUDA 12.8ビルドはRTX 50シリーズ（Blackwell）向けです。
ROS Noetic標準のPython 3.8とは分離し、Python 3.10環境向けにcv_bridgeを再ビルドします。

## 配置

次の4ファイルを `Yolov5_StrongSORT_OSNet` 直下へ置きます。

```text
Dockerfile.rtx5070
docker-entrypoint_rtx5070.sh
build_rtx5070.sh
run_rtx5070.sh
```

## 実行権限

```bash
chmod +x \
  docker-entrypoint_rtx5070.sh \
  build_rtx5070.sh \
  run_rtx5070.sh
```

## ビルド

```bash
sudo ./build_rtx5070.sh
```

直接ビルドする場合：

```bash
sudo docker build \
  --network=host \
  -f Dockerfile.rtx5070 \
  -t yolov5-strongsort:rtx5070-cu128 \
  .
```

完全再ビルド：

```bash
sudo docker build \
  --network=host \
  --no-cache \
  -f Dockerfile.rtx5070 \
  -t yolov5-strongsort:rtx5070-cu128 \
  .
```

## 起動

OpenCV画面表示を許可します。

```bash
xhost +local:docker
```

ROS MasterとRTX 5070 PCが `192.168.1.23` の場合：

```bash
sudo -E env \
  ROS_MASTER_URI=http://192.168.1.23:11311 \
  ROS_IP=192.168.1.23 \
  ./run_rtx5070.sh
```

ROS Masterが別PCの場合：

```bash
sudo -E env \
  ROS_MASTER_URI=http://ROS_MASTER_PCのIP:11311 \
  ROS_IP=RTX5070_PC自身のIP \
  ./run_rtx5070.sh
```

## コンテナ内の確認

```bash
source /opt/ros/noetic/setup.bash
source /opt/ros_py310/setup.bash
source /home/dars/catkin_ws/devel/setup.bash
```

Python確認：

```bash
which python3
python3 --version
```

期待値：

```text
/opt/conda/envs/rosgpu/bin/python3
Python 3.10.x
```

GPU確認：

```bash
python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

ROS/Python依存確認：

```bash
python3 - <<'PY'
import rospy
import cv2
import cv_bridge
import pyrealsense2
from torchreid.metrics.distance import compute_distance_matrix

print("ROS / cv_bridge / RealSense / torchreid OK")
PY
```

## 実行

このプログラムは `best2.pt` と `strong_sort.yaml` をカレントディレクトリ基準で参照するため、先に移動します。

```bash
cd /home/dars/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet
```

```bash
rosrun Yolov5_StrongSORT track_save_cpu.py --device 0
```

FP16を使用する場合：

```bash
rosrun Yolov5_StrongSORT track_save_cpu.py \
  --device 0 \
  --half
```

別ターミナルで確認：

```bash
watch -n 0.5 nvidia-smi
```

`Processes` に `Type C` の `python3` が表示されればGPU計算中です。

## 補足

ホスト側の `nvidia-smi` に表示されるCUDA Versionと、
PyTorchの `torch.version.cuda` は役割が異なります。

- `nvidia-smi`: ドライバが対応可能なCUDA上限
- `torch.version.cuda`: PyTorch wheelが同梱するCUDA runtime

RTX 5070版では `torch.version.cuda == 12.8` を期待します。
