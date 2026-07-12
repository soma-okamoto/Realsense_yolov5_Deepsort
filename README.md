# Docker環境のビルド・実行手順

本プロジェクトでは、使用するGPUに応じて以下のDocker環境を使い分けます。

* GTX 1650：CUDA 11.8版
* RTX 5070：CUDA 12.8版

## 1. プロジェクトディレクトリへ移動

```bash
cd ~/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet
```

## 2. シェルスクリプトへ実行権限を付与

初回のみ、各スクリプトへ実行権限を付与します。

```bash
chmod +x \
  docker-entrypoint.sh \
  build_gtx1650.sh \
  run_gtx1650.sh \
  docker-entrypoint_rtx5070.sh \
  build_rtx5070.sh \
  run_rtx5070.sh
```

---

# GTX 1650版

## 3. Dockerイメージのビルド

GTX 1650では、PyTorch 2.0.1およびCUDA 11.8の環境を使用します。

```bash
sudo docker build \
  --network=host \
  -f Dockerfile \
  -t yolov5-strongsort:gtx1650-cu118 \
  .
```

キャッシュを使用せず、完全に再ビルドする場合は以下を実行します。

```bash
sudo docker build \
  --network=host \
  --no-cache \
  -f Dockerfile \
  -t yolov5-strongsort:gtx1650-cu118 \
  .
```

## 4. コンテナの起動

OpenCVのウィンドウ表示を許可します。

```bash
xhost +local:docker
```

ROS MasterのIPアドレスと、このPC自身のIPアドレスを指定して起動します。

```bash
sudo -E env \
  ROS_MASTER_URI=http://192.168.1.23:11311 \
  ROS_IP=192.168.1.23 \
  ./run_gtx1650.sh
```

---

# RTX 5070版

## 5. Dockerイメージのビルド

RTX 5070では、PyTorch 2.7.1およびCUDA 12.8の環境を使用します。

```bash
sudo docker build \
  --network=host \
  -f Dockerfile.rtx5070 \
  -t yolov5-strongsort:rtx5070-cu128 \
  .
```

キャッシュを使用せず、完全に再ビルドする場合は以下を実行します。

```bash
sudo docker build \
  --network=host \
  --no-cache \
  -f Dockerfile.rtx5070 \
  -t yolov5-strongsort:rtx5070-cu128 \
  .
```

## 6. コンテナの起動

OpenCVのウィンドウ表示を許可します。

```bash
xhost +local:docker
```

ROS MasterのIPアドレスと、このPC自身のIPアドレスを指定して起動します。

```bash
sudo -E env \
  ROS_MASTER_URI=http://192.168.1.23:11311 \
  ROS_IP=192.168.1.23 \
  ./run_rtx5070.sh
```

---

# コンテナ内での準備

## 7. catkinワークスペースのビルド

必要に応じて、コンテナ内でcatkinワークスペースをビルドします。

```bash
cd /home/dars/catkin_ws
catkin build
```

ビルド済みのワークスペースをホストからマウントしている場合は、この処理を省略できることがあります。

## 8. ROS環境の読み込み

GTX 1650版では、以下を実行します。

```bash
source /opt/ros/noetic/setup.bash
source /home/dars/catkin_ws/devel/setup.bash
```

RTX 5070版では、Python 3.10用のROS環境も読み込みます。

```bash
source /opt/ros/noetic/setup.bash
source /opt/ros_py310/setup.bash
source /home/dars/catkin_ws/devel/setup.bash
```

## 9. 実行ディレクトリへ移動

モデルやStrongSORTの設定ファイルを相対パスで読み込むため、必ず以下のディレクトリへ移動します。

```bash
cd /home/dars/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet
```

## 10. YOLOv5＋StrongSORTの実行

GPU 0を指定してROSノードを起動します。

```bash
rosrun Yolov5_StrongSORT track_save_cpu.py --device 0
```

ファイル名は `track_save_cpu.py` ですが、`--device 0` を指定することでGPUを使用します。

## 11. GPU使用状況の確認

ホスト側の別ターミナルで以下を実行します。

```bash
watch -n 0.5 nvidia-smi
```

`Processes` 欄に、計算プロセスを示す `Type C` の `python3` が表示されていれば、GPUで実行されています。
