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


#realsense,YOLO->objectdetection

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5

camerainfo = CameraInfo()
color_intr = rs.intrinsics()
fps = 30.
delay = 1/fps*0.5
# depth_frame = Image()
# color_frame = Image()
pubdata = yolov5_data()   

class Yolov5Detector:
    def __init__(self):
        # self.pipeline = rs.pipeline()
        # self.config = rs.config()
        # self.pro = self.pipeline.start(self.config)   
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.mf = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub], 10, delay)
        self.mf.registerCallback(self.ImageCallback)
        self.bridge = CvBridge()
       

    def get_mid_pos(self, frame, box, depth_data, randnum):
        # print("get mid pos...")
        distance_list = []
        mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] 
        print("midpos" + str(mid_pos))
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) 
        #print(box,)
        for i in range(randnum):
            bias = random.randint(-min_val//4, min_val//4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
            cv2.drawMarker(frame, (int(mid_pos[1]), int(mid_pos[0])), (255, 0, 0))
            #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
            if dist:
                distance_list.append(dist)
        # if distance_list == []:
        #     print("zero!\n")
        #     return 0
        # else:
        # print("make list...")
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


    def dectshow(self, org_img, boxs, depth_data, color_intr):
        img = org_img.copy()
        # dep_img = depth_data.copy()
        # br = CvBridge() #12.4new
        # pubimage = br.cv2_to_imgmsg(img,encoding="passthrough") #12.4new
        # pubimage_depth = br.cv2_to_imgmsg(dep_img,encoding="passthrough") #12.4new
        detect =[]

        for box in boxs:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] 

            cv2.drawMarker(img, (int(mid_pos[0]), int(mid_pos[1])), (0, 0, 255))
            cv2.drawMarker(depth_data, (int(mid_pos[0]), int(mid_pos[1])), (0, 0, 255))
            dist = self.get_mid_pos(org_img, box, depth_data, 24)
            print("dist :::::" + str(dist))
            # dist = 1
            cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            print(depth_data)
            #12.2new
            w = box[2] - box[0]
            h = box[3] - box[1]

            detect.append(float(box[5])) #class
            detect.append(float(box[0])) #xmin
            detect.append(float(box[1])) #ymin
            detect.append(float(w))
            detect.append(float(h))
            detect.append(float(box[4])) #conf
            # print(detect)

        # pub = rospy.Publisher('/multi_command', Float32MultiArray, queue_size=10)
        pub = rospy.Publisher('/multi_command', yolov5_data, queue_size=10)
        # image_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10) #12.4new
        # dep_image_pub = rospy.Publisher('/camera/depth_registered/image_raw', Image, queue_size=10) #12.4new
        #array_forPublish = Float32MultiArray(data=detect)
        
        # pub.publish(array_forPublish)
        # image_pub.publish(pubimage) #12.4new
        # dep_image_pub.publish(pubimage_depth) #12.4new

        #pubdata.yolo_box_data = array_forPublish
        # pubdata.header.stamp = rospy.Time.now()
        #pub.publish(pubdata)

        cv2.imshow('dec_img', img)
        cv2.imshow("depth_img",depth_data)
        "cv2.waitKey(10) #new
        # key = cv2.waitKey(1)
        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()

        # else:
        #     rospy.spin()
        


    # def depth_callback(msg_depth):
    #     global depth_frame
    #     depth_frame = msg_depth

    # def color_callback(msg_color):
    #     global color_frame
    #     color_frame = msg_color

    def ImageCallback(self, depthdata, colordata):
        # global color_frame
        # global depth_frame
        # depth_frame = depthdata
        # color_frame = colordata
        # depth_image = np.asanyarray(depthdata)
        # color_image = np.asanyarray(colordata)
        # rospy.loginfo(color_frame)
        # pubdata.header.stamp = rospy.Time.now()
        pubdata.header.stamp = colordata.header.stamp
        try:
            color_image = self.bridge.imgmsg_to_cv2(colordata, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depthdata, '32FC1')

        except CvBridgeError as e:
            rospy.logger(e)

        #depth_image = np.asanyarray(depth_image)
        color_image = np.asanyarray(color_image)

        print(type(color_image))

        results = model(color_image)
        
        boxs= results.pandas().xyxy[0].values
        
        #12.12new
        self.camerainfo_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camerainfo_callback)
        
        # color_intr = rs.video_stream_profile(self.pro.get_stream(rs.stream.color)).get_intrinsics()
        #boxs = np.load('temp.npy',allow_pickle=True)
        self.dectshow(color_image, boxs, depth_image, color_intr)
        print(boxs)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.08), cv2.COLORMAP_JET)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

    def camerainfo_callback(self, msg_camerainfo):
        global color_intr
        color_intr.height =  msg_camerainfo.height
        color_intr.width = msg_camerainfo.width
        color_intr.fx = msg_camerainfo.K[0]
        color_intr.fy = msg_camerainfo.K[4]
        color_intr.ppx = msg_camerainfo.K[2]
        color_intr.ppy = msg_camerainfo.K[5]
        color_intr.model = rs.distortion.inverse_brown_conrady
        color_intr.coeffs = msg_camerainfo.D


    # class RealsenseCapture:

    #     def __init__(self):
    #         self.WIDTH = 640
    #         self.HEGIHT = 480
    #         self.FPS = 30
    #         # Configure depth and color streams
    #         self.config = rs.config()
    #         self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
    #         self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)

    #     def start(self):
    #         # Start streaming
    #         self.pipeline = rs.pipeline()
    #         pro = self.pipeline.start(self.config)
    #         print('pipline start')
    #         # Alignオブジェクト生成
    #         self.align_to = rs.stream.color
    #         self.align = rs.align(self.align_to)
    #         return pro

    #     def read(self, is_array=True):
    #         # Flag capture available
    #         ret = True
    #         # get frames
    #         frames = self.pipeline.wait_for_frames()

    #         aligned_frames = self.align.process(frames)

    #         # separate RGB and Depth image
    #         self.color_frame = aligned_frames.get_color_frame()  # RGB
    #         self.depth_frame = aligned_frames.get_depth_frame()  # Depth

    #         if not self.color_frame or not self.depth_frame:
    #             ret = False
    #             return ret, (None, None)
    #         elif is_array:
    #             # Convert images to numpy arrays
    #             color_image = np.array(self.color_frame.get_data())
    #             depth_image = np.array(self.depth_frame.get_data())
    #             return ret, (color_image, depth_image)
    #         else:
    #             return ret, (self.color_frame, self.depth_frame)

    #     def release(self):
    #         # Stop streaming
    #         self.pipeline.stop()


          
if __name__ == "__main__":
    rospy.init_node("ObjectDetection", anonymous=True)
    detector = Yolov5Detector()
    rospy.spin()
    # Configure depth and color streams
    # cap = RealsenseCapture()
    # # プロパティの設定
    # cap.WIDTH = 640
    # cap.HEIGHT = 480
    # cap.FPS = 30
    # # cv2.VideoCapture()と違ってcap.start()を忘れずに
    # pro = cap.start()
    # depth_frame = Image()
    # color_frame = Image()
    # global color_frame
    # global depth_frame
    # try:
    #     while True:
            # Wait for a coherent pair of frames: depth and color
            # ret,frames = cap.read()
            # depth_frame = frames[1]
            # color_frame = frames[0]
            # global color_frame
            # global depth_frame


            # depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
            # color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
            # mf = message_filters.ApproximateTimeSynchronizer([depth_sub, color_sub], 100, 10.0)
            # mf.registerCallback(ImageCallback)

            # depth_frame = Image()
            # color_frame = Image()

            # depth_image = np.asanyarray(depth_frame)

            # color_image = np.asanyarray(color_frame)
            #print(color_image)
            
            # color_image = bridge.imgmsg_to_cv2(color_frame, 'passthrough')
            # depth_image = bridge.imgmsg_to_cv2(depth_frame, 'passthrough')
            # color_image = bridge.imgmsg_to_cv2(color_frame, 'passthrough')

            # results = model(color_image)
            
            # boxs= results.pandas().xyxy[0].values
            
            # color_intr = rs.video_stream_profile(pro.get_stream(rs.stream.color)).get_intrinsics()


            
            

    #         #boxs = np.load('temp.npy',allow_pickle=True)
    #         dectshow(color_image, boxs, depth_image,color_intr)

    #         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    #         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
    #             depth_frame, alpha=0.08), cv2.COLORMAP_JET)
    #         # Stack both images horizontally
    #         # images = np.hstack((color_image, depth_colormap))
    #         # Show images
    #         #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #         #cv2.imshow('RealSense', images)
    #         key = cv2.waitKey(1)
    #         # Press esc or 'q' to close the image window
    #         if key & 0xFF == ord('q') or key == 27:
    #             cv2.destroyAllWindows()
    #             break
    # finally:
    #     # Stop streaming
    #     # cap.release()
    #     cv2.destroyAllWindows()
        
