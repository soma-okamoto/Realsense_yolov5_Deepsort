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

次の5ファイルを `Yolov5_StrongSORT_OSNet` 直下へ置きます。

```text
Dockerfile.rtx5070
docker-entrypoint_rtx5070.sh
rebuild_cv_bridge_rtx5070.sh
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

### ビルド時のエラー

`docker buildx build requires 1 argument` はビルドコンテキストの指定漏れです。
上記の直接ビルドコマンドでは、最後の `.` も含めて実行してください。
`sudo ./build_rtx5070.sh` にも `.` は指定済みです。

NVIDIA リポジトリへの接続で `Certificate verification failed` や
`The revocation or OCSP data are old and have been superseded` が出る場合、
その後の `does not have a Release file` は同じ接続失敗に伴うエラーです。
この Dockerfile は最初の `apt-get update` より前に、ベースイメージ由来の
`/etc/apt/sources.list.d/cuda*.list` を削除します。
CUDA/cuDNN はベースイメージに導入済みで、PyTorch は cu128 wheel を使用するため、
このビルドでは NVIDIA APT リポジトリは不要です。TLS 証明書検証は有効のままです。
修正後は `sudo ./build_rtx5070.sh` で再ビルドしてください（`--no-cache` は不要です）。

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
# コンテナ内で実行。ワークスペースのスクリプトを使うので再ビルド前でも適用できます。
cd /home/dars/catkin_ws/src/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet
bash ./docker-entrypoint_rtx5070.sh bash
```

起動スクリプトはROSとcatkin環境を読み込んだ後、Python 3.10用cv_bridgeを優先し、
システムのlibffi 7を `LD_PRELOAD` へ追加します。この後にホストの
`devel/setup.bash` を再度読み込むと、cv_bridgeの優先順位が変わる場合があります。

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

`libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`
が出る場合も、上記の起動スクリプトを通して新しいシェルに入り直してください。
`sudo docker exec -it ... bash` だけで入ったシェルには、コンテナ起動時に
エントリポイントが設定した `LD_PRELOAD` は引き継がれません。
修正済みスクリプトで起動した後、次の実変換テストで確認できます。

```bash
python3 - <<'PY'
import numpy as np
import cv_bridge

print("cv_bridge:", cv_bridge.__file__)
assert cv_bridge.__file__.startswith("/opt/ros_py310/"), cv_bridge.__file__
bridge = cv_bridge.CvBridge()
bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
msg = bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg, "rgb8"), bgr[:, :, ::-1])
depth = np.array([[1.25]], dtype=np.float32)
msg = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg, "passthrough"), depth)
print("cv_bridge color/depth conversion OK")
PY
```

### Python 3.8用Boostが読み込まれる場合

`libboost_python38.so.1.71.0: undefined symbol: _Py_fopen` は、
Python 3.10用のcv_bridgeがPython 3.8用Boostにリンクしていることを示します。
環境変数の変更だけでは修復できないため、cv_bridgeを再ビルドします。
このDockerfileは `rebuild_cv_bridge_rtx5070.sh` でBoostの `python310` 成分を明示し、
システムのBoostConfigによる別バージョンの選択を無効化しています。
再ビルド後には `ldd` によるリンク先の確認と、カラー・深度画像の実変換を行います。

現在のコンテナを修復する場合は、cv_bridgeを使うノードを終了し、
ホスト側の別ターミナルで実行します（コンテナ自体は起動したままにします）。

```bash
sudo docker exec -u 0 yolov5_strongsort_rtx5070 \
  bash /home/dars/catkin_ws/src/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet/rebuild_cv_bridge_rtx5070.sh
```

`cv_bridge Python 3.10 / Boost.Python 3.10 color/depth conversion OK` が出たら、
コンテナ内で修正済みエントリポイントを通してシェルを開き、ノードを起動し直します。
この修復はコンテナ内への変更なので、コンテナを作り直す前に
`sudo ./build_rtx5070.sh` で修正済みイメージもビルドしてください。

## 実行

### QR位置とMeta画像を使用する場合

`QRPostion_test.py` の姿勢変換は、Python 3.10環境に導入済みのSciPyを使用します。
このノードのために `tf.transformations` を追加インストールする必要はありません。
ソースはホストのワークスペースからマウントされるため、このPythonファイルの修正だけなら
イメージの再ビルドは不要です。実行中のノードを終了してから起動し直してください。

コンテナ内でQR検出ノードを起動します。

```bash
rosrun Yolov5_StrongSORT QRPostion_test.py
```

このノードにはRealSenseのカラー画像・対応する深度画像・CameraInfo、および
`arm_2/arm_controller/position_command` の関節角（5関節）が必要です。
`Main` で始まるQRを検出し、有効な深度と関節角を取得すると `/QRposition` を配信します。

ホストの別ターミナルから、同じコンテナに入ります。

```bash
sudo docker exec -it yolov5_strongsort_rtx5070 \
  bash /home/dars/catkin_ws/src/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet/docker-entrypoint_rtx5070.sh bash
```

Meta側で `/meta_color_pose_qr`（`geometry_msgs/PoseStamped`）も配信したうえで、
次のノードを起動します。`QRPostion_test.py` はこのMeta側トピックを配信しません。

```bash
rostopic info /QRposition
rostopic info /meta_color_pose_qr
rosrun Yolov5_StrongSORT track_P_meta.py
```

`track_P_meta.py` はカラー画像・深度画像・`/meta_color_pose_qr` を時刻同期するため、
Meta側の姿勢トピックが未配信のままだと画像処理のコールバックも呼ばれません。

### track_save_cpu.pyを使用する場合

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
