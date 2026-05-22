#!/usr/bin/env python3 
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray

def box_callback(msg):
    global boxPosition
    #rospy.loginfo("Message '{}' recieved".format(msg))
    boxPosition = msg

def callback(data):
    points = []
    resolution = (data.height, data.width)
    for p in pc2.read_points(data, skip_nans=False, field_names=("x", "y", "z")):
        points.append(p[2])
    if len(points) == data.width*data.height:
        z_points = np.array(points, dtype=np.float32)
        print(len(z_points))
        z = z_points.reshape(resolution)
        print("z : " ,z)
        print("z count :",len(z[0]))

        if not (boxPosition.data[0]==0 and boxPosition.data[2]==0):
            print('Box: {}, {}'.format(boxPosition.data[0], boxPosition.data[2]))
            print("z : " ,z)
            print("z_width : ",len(z[0]))
            
            z_box = z[0][boxPosition.data[0]:boxPosition.data[2]]
            
            print(z_box)
            z_value = z_box[~np.isnan(z_box)]

            distance = min(z_value)
            print('Distance: {}'.format(distance))
        # print(len(x))
        # print(len(y))
        # print(len(z))
        # bounding_box = [0,0,200,200]
        # print(boxPosition.data)
        # mask = (x >= boxPosition.data[0]) & (x <= boxPosition.data[2]) & (y >= boxPosition.data[1]) &(y <= boxPosition.data[3])
        # print(len(mask))
        # x = x[mask]
        # y = y[mask]
        # z = z[mask]
        
        # print ("x : ",x )
        # print ("y : ",y )
        # print ("z : ",z )



if __name__ == "__main__":
    rospy.sleep(1)
    rospy.init_node("pointcloud_subscriber")

    rospy.Subscriber("/camera/depth/color/points",PointCloud2,callback)
    box_sub = rospy.Subscriber("/box_command", Float32MultiArray, box_callback)
    boxPosition = Float32MultiArray()
    rospy.spin()








# class PointCloudConverter:
#     def __init__(self):
#         self.point_cloud = None
#         self.height = None
#         rospy.Subscriber("/camera/depth/color/points",PointCloud2,self.callback)
    
#     def callback(self,msg):
#         self.height = msg.height
#         self.point_cloud = np.array(list(pc2.read_points(msg,skip_nans=True,field_names=("x","y","z"))))

    
#     def get_world_coordinates(self,screen_x,screen_y):
#         if self.point_cloud is None:
#             return None
        
#         index_x = int(screen_x)
#         index_y = int(screen_y)
#         print("pointcloud_x : " + str(self.point_cloud.shape[1]) + ":pointcloud_y : " + str(self.point_cloud.shape[0]))
#         print("index_x :" + str(index_x) + ":index_y : " + str(index_y))
#         print(self.height)
#         world_x  = self.point_cloud[index_y *self.height + index_x ][0]
#         world_y  = self.point_cloud[index_y *self.height + index_x ][1]
#         world_z  = self.point_cloud[index_y *self.height + index_x ][2]


#         return ( world_x,world_y,world_z)

# if __name__ == "__main__":
#     rospy.init_node("point_cloud_converter")
#     converter = PointCloudConverter()

#     screen_x = 100
#     screen_y = 100
#     while not rospy.is_shutdown():
#         world_coordinates = converter.get_world_coordinates(screen_x,screen_y)

#         if(world_coordinates is not None):
#             print("World coordinates for screeen")
#         else:
#             print("Point cloud data available")


# #!/usr/bin/env python3 
# import time, os, sys, signal, threading 
 
# import numpy as np 
# from brics_actuator.msg import JointPositions
# from std_msgs.msg import Float32MultiArray
# import rospy 
# from sensor_msgs.msg import PointCloud2 
# from sensor_msgs import point_cloud2
# import time
# from geometry_msgs.msg import PoseStamped,PointStamped
# import tf
# import pyrealsense2 as rs
# import open3d as o3d


# def box_callback(msg):
#     global boxPosition
#     #rospy.loginfo("Message '{}' recieved".format(msg))
#     boxPosition = msg

# def callback_pointcloud(data):
#     global pointcloud
#     pointcloud = data

# point_cloud = rs.pointcloud()


# if __name__ == '__main__':
#     rospy.init_node("pcl_listener" , anonymous=True)
#     cameraPosition_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2,callback_pointcloud)
#     pointcloud = PointCloud2()
#     box_sub = rospy.Subscriber("/box_command", Float32MultiArray, box_callback)
#     boxPosition = Float32MultiArray()

#     rate = rospy.Rate(10)

#     while not rospy.is_shutdown():
#         points = point_cloud.calculate(pointcloud)
