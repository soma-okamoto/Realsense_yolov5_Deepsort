# GTX 1650用 YOLOv5 + StrongSORT + ROS Noetic Docker

## 採用構成

- ROS Noetic / Ubuntu 20.04 / Python 3.8
- PyTorch 2.0.1 + torchvision 0.15.2
- CUDA 11.8版PyTorch wheel
- GTX 1650の4 GB VRAMを想定
- pyrealsense2 2.50.0.3812（Python 3.8対応）

RTX 5070向けのCUDA 12.8構成とは別イメージです。

## 1. ファイル配置

この4ファイルを次のディレクトリへ置きます。

```text
Dockerfile
docker-entrypoint.sh
.dockerignore
build_gtx1650.sh
run_gtx1650.sh
```

配置先：

```bash
cd ~/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet
```

この位置に次のファイルが存在することも確認します。

```text
requirements.txt
yolov5/requirements.txt
track_save_cpu.py
```

## 2. ホスト側のGPU確認

```bash
nvidia-smi
```

続けてDockerからGPUが見えるか確認します。

```bash
sudo docker run --rm --gpus all ubuntu:20.04 nvidia-smi
```

これが失敗する場合は、Dockerfileではなくホスト側のNVIDIAドライバまたはNVIDIA Container Toolkitの問題です。

## 3. ビルド

```bash
chmod +x build_gtx1650.sh run_gtx1650.sh docker-entrypoint.sh
sudo ./build_gtx1650.sh
```

スクリプトを使わない場合：

```bash
sudo docker build -t yolov5-strongsort:gtx1650-cu118 .
```

## 4. 起動

ROS MasterとこのPCがどちらも `192.168.1.23` の場合：

```bash
sudo -E ./run_gtx1650.sh
```

ROS MasterとこのPCのIPが異なる場合：

```bash
sudo -E \
  ROS_MASTER_URI=http://192.168.1.23:11311 \
  ROS_IP=このGTX1650_PCのIP \
  ./run_gtx1650.sh
```

`ROS_IP` はROS MasterのIPではなく、このコンテナを動かしているPC自身のIPです。

画面表示で `cannot open display` が出た場合は、ホストで一度だけ次を実行してから起動します。

```bash
xhost +local:docker
sudo -E ./run_gtx1650.sh
```

## 5. コンテナ内でGPU確認

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else 0)
PY
```

期待値：

```text
torch: 2.0.1+cu118
CUDA runtime: 11.8
CUDA available: True
GPU: NVIDIA GeForce GTX 1650
```

## 6. ROSノードをGPUで実行

`track_save_cpu.py` は `--device` 引数を持っているため、ファイル名を変えなくてもGPUを選択できます。

まず標準設定：

```bash
rosrun Yolov5_StrongSORT track_save_cpu.py --device 0
```

GTX 1650でCUDAメモリ不足が出る場合：

```bash
rosrun Yolov5_StrongSORT track_save_cpu.py --device 0 --imgsz 416
```

さらにFP16を試す場合：

```bash
rosrun Yolov5_StrongSORT track_save_cpu.py --device 0 --imgsz 416 --half
```

ただし、このコードは内部の `letterbox()` を引数なしで呼んでおり、実際には640×640の前処理が固定されている版があります。その場合、`--imgsz 416` だけでは入力サイズが下がりません。メモリ不足が発生した段階で、次の行を修正します。

変更前：

```python
img = letterbox(im0s)[0]
```

変更後：

```python
img = letterbox(im0s, new_shape=imgsz, stride=stride)[0]
```

## 7. 実行中に本当にGPUを使っているか確認

別ターミナルで：

```bash
watch -n 1 nvidia-smi
```

`python3` プロセスとGPUメモリ使用量が表示されればGPU実行されています。

## よくあるエラー

### `could not select device '0'`

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

`False` ならホスト側ドライバまたはNVIDIA Container Toolkitを確認します。

### `CUDA out of memory`

最初に `--imgsz 416`、次に `--half` を試します。上記の `letterbox()` 修正も確認してください。

### `ModuleNotFoundError: yolov5_ros.msg`

ワークスペースが未ビルド、またはメッセージ生成前です。

```bash
cd /home/dars/catkin_ws
catkin build
source devel/setup.bash
```

### `ModuleNotFoundError: pyrealsense2`

Dockerfileでは `pyrealsense2` をインストール済みです。再ビルド時に古いキャッシュを避けるには：

```bash
sudo docker build --no-cache -t yolov5-strongsort:gtx1650-cu118 .
```
