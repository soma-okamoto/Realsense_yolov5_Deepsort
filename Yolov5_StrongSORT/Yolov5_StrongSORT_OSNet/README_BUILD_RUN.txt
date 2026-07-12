1) Put Dockerfile and docker-entrypoint.sh in:
   ~/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet/

2) Build from that directory:
   cd ~/catkin_ws/src/Realsense_yolov5_Deepsort/Yolov5_StrongSORT/Yolov5_StrongSORT_OSNet
   sudo docker build -t yolov5-strongsort-gpu .

3) Start the container (replace IPs if necessary):
   sudo docker rm -f yolov5_strongsort_gpu 2>/dev/null || true
   sudo docker run -it \
     --name yolov5_strongsort_gpu \
     --gpus all \
     --network=host \
     --ipc=host \
     -v ~/catkin_ws:/home/dars/catkin_ws \
     -e ROS_MASTER_URI=http://192.168.1.23:11311 \
     -e ROS_IP=192.168.1.23 \
     yolov5-strongsort-gpu

4) Verify CUDA:
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

5) Run:
   rosrun Yolov5_StrongSORT track_save_cpu.py --device 0

If the script does not define --device, change its hard-coded device from cpu to 0/cuda:0.
The first line of track_save_cpu.py should preferably be:
   #!/usr/bin/env python3
