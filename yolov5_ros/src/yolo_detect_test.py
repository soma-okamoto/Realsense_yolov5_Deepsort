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


#realsense,YOLO->objectdetection

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5

def get_mid_pos(frame,box,depth_data,randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] 


    ###
    ###
    mid_pos1 = [(box[0] + box[2])*3//2, (box[1] + box[3])//2]
    mid_pos2 = [-(box[0] + box[2])//2, (box[1] + box[3])//2]

    if(mid_pos1[0] >630):
        mid_pos1[0] = 630
    if(mid_pos1[1] > 470):
        mid_pos1[1] = 470

    if(mid_pos2[0] >630):
        mid_pos2[0] = 630
    if(mid_pos2[1] > 470):
        mid_pos2[1] = 470
+

    print(mid_pos1)
    print(mid_pos2)
    ###
    ###


    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) 
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        

        ###
        ###
        dist1 = depth_data[int(mid_pos1[1] + bias), int(mid_pos1[0] + bias)]
        dist2 = depth_data[int(mid_pos2[1] + bias), int(mid_pos2[0] + bias)]
        ###
        ###


        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)

        ###
        ###
        if dist1:
            distance_list.append(dist1)
        if dist2:
            distance_list.append(dist2)
        ###
        ###

    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] 
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)


    
def dectshow(org_img, boxs,depth_data,color_intr):
    img = org_img.copy()
    detect =[]
    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        ###
        ###
        #右
        if box[5] == "bottle" and 3.5 < ((box[3]-box[1])/(box[2]-box[0]) and dist1 > dist):
            box[2] = int(box[0] + (box[3] - box[1]/3.3))
        elif box[5] == "bottle" and 6.4 < ((box[3]-box[1])/(box[2]-box[0]) and dist1 > dist):
            box[2] = int(box[0] + (box[3] - box[1]/3.3))
            dist = dist1 + 0.87*(box[2] - box[0])/2

        #左
        if box[5] == "bottle" and 3.5 < ((box[3]-box[1])/(box[2]-box[0]) and dist2 < dist):
            box[0] = int(box[2] - (box[3] - box[1]/3.3))
        elif box[5] == "bottle" and 6.4 < ((box[3]-box[1])/(box[2]-box[0]) and dist2 < dist):
            box[0] = int(box[2] - (box[3] - box[1]/3.3))
            dist = dist2 + 0.87*(box[2] - box[0])/2

        #下
        if box[5] == "bottle" and 2.8 > ((box[3]-box[1])/(box[2]-box[0])):
            box[3] = int(box[1] + (box[2] - box[0])*3.4)
        ###
        ###
       
        dist = get_mid_pos(org_img, box, depth_data, 24)

        ###
        ###
        dist1 = get_mid_pos(org_img, box, depth_data, 24)
        dist2 = get_mid_pos(org_img, box, depth_data, 24)
        ###
        ###

        cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        x = box[0] + (box[2] - box[0])/2
        y = box[1] + (box[3] - box[1])/2
        world_point = rs.rs2_deproject_pixel_to_point(color_intr , [x,y],dist/1000)
        
        #eef（グリッパー）から見たobject位置
        from_eef_to_object=np.array([[-world_point[1]],[world_point[0]],[dist/1000]])
        #from_eef_to_object=np.array([[0],[0],[0]])
        #関節角状態
        q5= 0
        q4= -100
        q3= -70
        q2= 70
        q1= -60


        rot_camera=rot_y(DegToRad(-8))
        trans5=np.array([[0.06],[0.033],[0.08]])
        rot_5=rot_z(DegToRad(q5))

        trans4=np.array([[0],[0],[0.13]])
        rot_4=rot_y(DegToRad(q4))

        trans3=np.array([[0],[0],[0.135]])
        rot_3=rot_y(DegToRad(q3))

        trans2=np.array([[0],[0],[0.155]])
        rot_2=rot_y(DegToRad(q2))

        trans1=np.array([[0.033],[0],[0.019]])
        rot_1=rot_z(DegToRad(q1))
        #0.091は地面からbase_linkへの並進移動
        trans0=np.array([[-0.147],[0],[0.152+0.091]])
        rot_0=rot_z(DegToRad(180))

        #baseから見たobject
        object2base=trans0+rot_0@(rot_1@(trans1+rot_2@(trans2+rot_3@(trans3+rot_4@(trans4+rot_5@(rot_camera@(trans5+from_eef_to_object)))))))


        """
        detect.append(float(box[0]))
        detect.append(float(box[1]))
        detect.append(float(box[2]))
        detect.append(float(box[3]))
        detect.append(float(box[4]))
        detect.append(float(box[5]))
        detect.append(float(dist/1000))
        
        detect.append(float(world_point[0]))
        detect.append(float(world_point[1]))
        detect.append(float(dist/1000))
        detect.append(float(box[5]))
        """

        detect.append(float(object2base[0][0]))
        detect.append(float(object2base[1][0]))
        detect.append(float(object2base[2][0]))
        detect.append(float(box[5]))
        print(detect)
    
    if len(boxs) >=2:

        Point_I = rs.rs2_deproject_pixel_to_point(color_intr , [143,451],0.5)
        #ロの3次元座標推定

        Point_R = rs.rs2_deproject_pixel_to_point(color_intr , [490,451],0.5)

        est_range = math.sqrt((Point_I[0]-Point_R[0])*(Point_I[0]-Point_R[0]) + (Point_I[1]-Point_R[1])*(Point_I[1]-Point_R[1]) +(Point_I[2]-Point_R[2])*(Point_I[2]-Point_R[2]))
        #print(est_range)

    pub = rospy.Publisher('/multi_command', Float32MultiArray, queue_size=10)
    array_forPublish = Float32MultiArray(data=detect)
    
    pub.publish(array_forPublish)

    cv2.imshow('dec_img', img)



class RealsenseCapture:

    def __init__(self):
        self.WIDTH = 640
        self.HEGIHT = 480
        self.FPS = 60
        # Configure depth and color streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.WIDTH, self.HEGIHT, rs.format.bgr8, self.FPS)
        self.config.enable_stream(rs.stream.depth, self.WIDTH, self.HEGIHT, rs.format.z16, self.FPS)

    def start(self):
        # Start streaming
        self.pipeline = rs.pipeline()
        pro = self.pipeline.start(self.config)
        print('pipline start')
        # Alignオブジェクト生成
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        return pro

    def read(self, is_array=True):
        # Flag capture available
        ret = True
        # get frames
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        # separate RGB and Depth image
        self.color_frame = aligned_frames.get_color_frame()  # RGB
        self.depth_frame = aligned_frames.get_depth_frame()  # Depth

        if not self.color_frame or not self.depth_frame:
            ret = False
            return ret, (None, None)
        elif is_array:
            # Convert images to numpy arrays
            color_image = np.array(self.color_frame.get_data())
            depth_image = np.array(self.depth_frame.get_data())
            return ret, (color_image, depth_image)
        else:
            return ret, (self.color_frame, self.depth_frame)

    def release(self):
        # Stop streaming
        self.pipeline.stop()


#affine

#eef（グリッパー）から見たobject位置を入力するとbase_footprintに変換される

def DegToRad(th):
  rad=(np.pi/180)*th
  return rad

# 回転行列
def rot_x(th):
  
  c = np.cos(th)
  s = np.sin(th)
  
  Rx = np.array([[1, 0, 0],
               [0, c, -s],
               [0, s, c]])
  
  return Rx

def rot_y(th):
  
  c = np.cos(th)
  s = np.sin(th)
  
  Ry = np.array([[c, 0, s],
               [0, 1, 0],
               [-s, 0, c]])

  return Ry

def rot_z(th):
  
  c = np.cos(th)
  s = np.sin(th)
  
  Rz = np.array([[c, -s, 0],
               [s, c, 0],
               [0, 0, 1]])  
  
  return Rz
  

        
if __name__ == "__main__":
    rospy.init_node("ObjectDetection", anonymous=True)
    # Configure depth and color streams
    cap = RealsenseCapture()
    # プロパティの設定
    cap.WIDTH = 640
    cap.HEIGHT = 480
    cap.FPS = 60
    # cv2.VideoCapture()と違ってcap.start()を忘れずに
    pro = cap.start()
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            ret,frames = cap.read()
            depth_frame = frames[1]
            color_frame = frames[0]
            
            depth_image = np.asanyarray(depth_frame)

            color_image = np.asanyarray(color_frame)

            results = model(color_image)
            
            boxs= results.pandas().xyxy[0].values
            
            color_intr = rs.video_stream_profile(pro.get_stream(rs.stream.color)).get_intrinsics()


            
            

            #boxs = np.load('temp.npy',allow_pickle=True)
            dectshow(color_image, boxs, depth_image,color_intr)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_frame, alpha=0.08), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        cap.release()
        cv2.destroyAllWindows()
        
