#!/usr/bin/env python3
from __future__ import print_function
from cProfile import label
import profile
import rospy
from std_msgs.msg import Float32MultiArray
import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import math
from sensor_msgs.msg import Image #12.4new
from cv_bridge import CvBridge, CvBridgeError #12.4new
import message_filters
from sensor_msgs.msg import CameraInfo
from yolov5_ros.msg import yolov5_data

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5

camerainfo = CameraInfo()
pubdata = yolov5_data()

fps = 30.
delay = 1/fps*0.5

def starttest():
    print("setup now...")
    # time.sleep(5)
    
def ImageCallback(depthdata, colordata):
    global color_frame
    global depth_frame
    color_frame = colordata
    depth_frame = depthdata

def camerainfo_callback(msg_camerainfo):
    global color_intr
    color_intr.height =  msg_camerainfo.height
    color_intr.width = msg_camerainfo.width
    color_intr.fx = msg_camerainfo.K[0]
    color_intr.fy = msg_camerainfo.K[4]
    color_intr.ppx = msg_camerainfo.K[2]
    color_intr.ppy = msg_camerainfo.K[5]
    color_intr.model = rs.distortion.inverse_brown_conrady
    color_intr.coeffs = msg_camerainfo.D

def get_mid_pos(frame, box, depth_data, randnum):
    # print("get mid pos...")
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] 
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) 
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)

    distance_list = np.array(distance_list)
    # print("sort...")
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] 
    #print(distance_list, np.mean(distance_list))
    # print("distance list :", distance_list)
    if distance_list.size > 0:
        return np.mean(distance_list)

    else:
        # print("zero!\n")
        return 0   

def dectshow(org_img, boxs, depth_data, color_intr):
    img = org_img.copy()
    # dep_img = depth_data.copy()
    # br = CvBridge() #12.4new
    # pubimage = br.cv2_to_imgmsg(img,encoding="passthrough") #12.4new
    # pubimage_depth = br.cv2_to_imgmsg(dep_img,encoding="passthrough") #12.4new
    detect =[]

    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        dist = get_mid_pos(org_img, box, depth_data, 24)
        # dist = 1
        cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #12.2new
        w = box[2] - box[0]
        h = box[3] - box[1]

        detect.append(float(box[5])) #class
        detect.append(float(box[0])) #xmin
        detect.append(float(box[1])) #ymin
        detect.append(float(w))
        detect.append(float(h))
        detect.append(float(box[4])) #conf
        detect.append(float(dist)/1000)
        # print(detect)

    return img, detect

def main():
    global color_frame
    global depth_frame

    rospy.init_node("ObjectDetection", anonymous=True)
    bridge = CvBridge()

    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    mf = message_filters.ApproximateTimeSynchronizer([depth_sub, color_sub], 10, delay)
    mf.registerCallback(ImageCallback)

    camerainfo_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, camerainfo_callback)

    try:
        while True:
            # print(color_frame)
            # print(color_frame.header.seq)
            if color_frame.header.seq != 0:
                # print("cvbrige start")
                try:
                    color_image = bridge.imgmsg_to_cv2(color_frame, 'bgr8')
                    depth_image = bridge.imgmsg_to_cv2(depth_frame, '32FC1')

                except CvBridgeError as e:
                    rospy.logger(e)

                pubdata.header.stamp = color_frame.header.stamp
                results = model(color_image)
                boxs= results.pandas().xyxy[0].values

                show_img, detect_data = dectshow(color_image, boxs, depth_image, color_intr)

                pub = rospy.Publisher('/multi_box_command', yolov5_data, queue_size=10)
                array_forPublish = Float32MultiArray(data=detect_data)
                pubdata.yolo_box_data = array_forPublish
                pub.publish(pubdata)
                
                # print("image show")
                # rospy.loginfo(color_img)
                cv2.imshow('dec_img', show_img)
                key = cv2.waitKey(1) #new
                # print("wait message")
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

            # rospy.sleep(0.2)

    finally:
        # Stop streaming
        cv2.destroyAllWindows()

if __name__ == '__main__':
    starttest()
    color_frame = Image()
    depth_frame = Image()
    color_intr = rs.intrinsics()
    main()
        
