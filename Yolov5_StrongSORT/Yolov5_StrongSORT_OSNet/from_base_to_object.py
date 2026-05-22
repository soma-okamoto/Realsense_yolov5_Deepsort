#!/usr/bin/env python3
import pyrealsense2 as rs
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from brics_actuator.msg import JointPositions
from std_msgs.msg import Float32MultiArray
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped,PointStamped,Point
import tf
import geometry_msgs
import open3d as o3d


def box_callback(msg):
    global boxPosition
    #rospy.loginfo("Message '{}' recieved".format(msg))
    boxPosition = msg

def quaternion_to_euler(quaternion):
    """Convert Quaternion to Euler Angles

    quarternion: geometry_msgs/Quaternion
    euler: geometry_msgs/Vector3
    """

    e = tf.transformations.euler_from_quaternion((quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
    euler = [e[0],e[1],e[2]]

    return euler

def transform_position(ps_list, source_link='base_link', target_link='camera_link'):
    listener = tf.TransformListener()
    ps = PoseStamped() 
    ps.header.frame_id = target_link
    ps.header.stamp = rospy.Time()
    ps.pose.position.x = ps_list[1] # カメラローカルのx座標
    ps.pose.position.y = ps_list[3] # カメラローカルのy座標
    ps.pose.position.z = ps_list[2] # カメラローカルのz座標
    #print(ps)
    try:
        listener.waitForTransform(source_link, target_link, rospy.Time(), rospy.Duration(10))
        tf_ps = listener.lookupTransform(source_link,target_link,rospy.Time(0))
        
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        raise e
    return tf_ps # PointStamped型で返却されることに注意


def transform_coordinate(x,y,z):

    try:

        trans = tfBuffer.lookup_transform("base_link","camera_color_optical_frame",rospy.Time(0))
        xyz_coordinate = geometry_msgs.msg.PointStamped()
        xyz_coordinate.header.frame_id = "camera_color_optical_frame"
        xyz_coordinate.header.stamp = rospy.Time()
        xyz_coordinate.point.x = x
        xyz_coordinate.point.y = y
        xyz_coordinate.point.z = z
        xyz_coordinate_in_base_link = tf2_geometry_msgs.do_transform_point(xyz_coordinate, trans)
        rospy.loginfo("xyz coordinate in base_link: (%.4f, %.4f, %.4f)" % (xyz_coordinate_in_base_link.point.x,
                                                                            xyz_coordinate_in_base_link.point.y,
                                                                            xyz_coordinate_in_base_link.point.z))
        return [xyz_coordinate_in_base_link.point.x,xyz_coordinate_in_base_link.point.y,xyz_coordinate_in_base_link.point.z]

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Failed to transform xyz coordinate")
        return None
    #transformed_coordinate = tf2_ros.transformations.transform_point(transform,x,y,z)

    
if __name__ == "__main__":
    rospy.init_node("get_object_position", anonymous=True)
    rate = rospy.Rate(10)
    box_sub = rospy.Subscriber("/box_command", Float32MultiArray, box_callback)
    boxPosition = Float32MultiArray()
    print(boxPosition)
    #cameraPosition_sub = rospy.Subscriber("arm_2/arm_controller/position_command", JointPositions, position_callback)
    #cameraPosition = JointPositions()
    #cameraInfo_sub = rospy.Subscriber("/camera/color/camera_info",CameraInfo,camerainfo_callback)
    #camerainfo = CameraInfo()

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    # while not rospy.is_shutdown():
    rospy.sleep(0.2)
    print(boxPosition)
    test = transform_coordinate(boxPosition.data[0],boxPosition.data[1],boxPosition.data[2])

    #pos_x,pos_y,pos_z = MarkerPosition.pose.position.x,MarkerPosition.pose.position.y,MarkerPosition.pose.position.z
    # #ori_x,ori_y,ori_z,ori_w = MarkerPosition.pose.orientation.x,MarkerPosition.pose.orientation.y,MarkerPosition.pose.orientation.z,MarkerPosition.pose.orientation.w
    # print(boxPosition.data)
    # h, w = 480,640
    # open3d_center_w, open3d_center_h, margin = w/2, h/2, 5
    # yolo_obj_x1, yolo_obj_y1 = (boxPosition.data[1]-open3d_center_w-margin)/1000, (boxPosition.data[2]-open3d_center_h-margin)/1000
    # yolo_obj_x2, yolo_obj_y2 = (boxPosition.data[3]-open3d_center_w-margin)/1000, (boxPosition.data[4]-open3d_center_h-margin)/1000
    # print("yolo_obj_x1 :::: " + str(yolo_obj_x1))
    # print("yolo_obj_y1 :::: " + str(yolo_obj_y1))        
    # print("yolo_obj_x2 :::: " + str(yolo_obj_x2))
    # print("yolo_obj_y2 :::: " + str(yolo_obj_y2))


    # bb = o3d.geometry.AxisAlignedBoundingBox(
    #     np.array([[yolo_obj_x1],[yolo_obj_y1],[-1.0]]),
    #     np.array([[yolo_obj_x2],[yolo_obj_y2],[1.0]])
    # )

    # # ======================= open3d ========================= #

    # pcd = o3d.io.read_point_cloud("./output.ply")
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # crop_pcd = pcd.crop(bb)

    # # Flip the pointclouds, otherwise they will be upside down.
    # o3d.visualization.draw([crop_pcd])

    # object = tf2_geometry_msgs.PoseStamped()
    # object.header.frame_id = "camera_link"
    
    # object.header.stamp = rospy.Time.now()
    # object.pose.position.x,object.pose.position.y,object.pose.position.z = boxPosition.data[0],boxPosition.data[1],boxPosition.data[2]
    # # print(boxPosition.data)
    # objectPosition = rospy.Publisher("box_transform",PoseStamped,queue_size = 5)
    # print(object)
    
    # objectTransform = PoseStamped()
    # #tf_ps = transform_position(boxPosition.data)
    # #print(tf_ps[0])
    # try:
    #     global_pose = tfBuffer.lookup_transform("base_link","camera_link",rospy.Time(0))

    #     quaternion = [global_pose.transform.rotation.x,global_pose.transform.rotation.y,global_pose.transform.rotation.z,global_pose.transform.rotation.w]
    #     euler = quaternion_to_euler(quaternion)

    #     #print(euler)
    #     #print(global_pose)
    #     objectTransform.pose.position.x,objectTransform.pose.position.y,objectTransform.pose.position.z = global_pose.transform.translation.x,global_pose.transform.translation.y,global_pose.transform.translation.z
    #     objectTransform.pose.orientation.x,objectTransform.pose.orientation.y,objectTransform.pose.orientation.z,objectTransform.pose.orientation.w = global_pose.transform.rotation.x,global_pose.transform.rotation.y,global_pose.transform.rotation.z,global_pose.transform.rotation.w
        
    #     objectPosition.publish(objectTransform)
    #     #object2base = CoordinateTransformation(global_pose.pose.position.x,global_pose.pose.position.y,global_pose.pose.position.z,color_intr)
    #     #print(pos_x,pos_y,pos_z)
    #     #print(object2base)
    # except(tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException):
    #     rospy.logwarn("tf not found")
    rate.sleep()