
```bash
cd Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet


chmod +x \
  docker-entrypoint.sh \
  build_gtx1650.sh \
  run_gtx1650.sh \
  docker-entrypoint_rtx5070.sh \
  build_rtx5070.sh \
  run_rtx5070.sh

```




```bash
sudo docker build \
  --network=host \
  -f Dockerfile \
  -t yolov5-strongsort:gtx1650-cu118 \
  .
```


```bash
sudo docker build \
  --network=host \
  --no-cache \
  -f Dockerfile.rtx5070 \
  -t yolov5-strongsort:rtx5070-cu128 \
  .
```

```bash
sudo -E env \
  ROS_MASTER_URI=http://192.168.1.23:11311 \
  ROS_IP=192.168.1.23 \
  ./run_gtx1650.sh
```

```bash
sudo -E env \
  ROS_MASTER_URI=http://192.168.1.23:11311 \
  ROS_IP=192.168.1.23 \
  ./run_rtx5070.sh
```

```bash
catkin build
```

```bash
source /opt/ros/noetic/setup.bash
source /home/dars/catkin_ws/devel/setup.bash
cd /home/dars/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet

rosrun Yolov5_StrongSORT track_save_cpu.py --device 0

```