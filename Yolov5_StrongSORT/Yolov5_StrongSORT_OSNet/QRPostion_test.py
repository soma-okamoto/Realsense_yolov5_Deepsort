#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import message_filters

from brics_actuator.msg import JointPositions
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft

QR_SIZE_M = 0.2


frame_count = 0
detect_interval = 1

last_points = None
last_data = ""


color_frame = None
depth_frame = None
cameraPosition = JointPositions()

color_intr = rs.intrinsics()
bridge = CvBridge()

qr_detector = cv2.QRCodeDetector()


def DegToRad(th):
    return (np.pi / 180.0) * th


def rot_x(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(th):
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def CoordinateTransformation(x, y, z, color_intr):
    """
    x, y: QR中心の画像座標[pixel]
    z: depth[m]
    return: base座標系から見たQR位置 3x1
    """

    world_point = rs.rs2_deproject_pixel_to_point(color_intr, [x, y], z)

    # 既存コードと同じ変換
    # RealSense camera座標 -> eef基準のobject座標っぽく変換
    from_eef_to_object = np.array([
        [-world_point[1]],
        [ world_point[0]],
        [ z]
    ])

    if len(cameraPosition.positions) < 5:
        rospy.logwarn("JointPositions がまだ取得できていません")
        return None

    test1 = -(cameraPosition.positions[0].value - DegToRad(169))
    test2 =  (cameraPosition.positions[1].value - DegToRad(65))
    test3 =  (cameraPosition.positions[2].value + DegToRad(146))
    test4 =  (cameraPosition.positions[3].value - DegToRad(102.5))
    test5 = -(cameraPosition.positions[4].value - DegToRad(167.5))

    rot_camera = rot_y(DegToRad(-8))
    trans5 = np.array([[0.055], [-0.0332], [0.06]])
    rot_5 = rot_z(test5)

    trans4 = np.array([[0], [0], [0.13]])
    rot_4 = rot_y(test4)

    trans3 = np.array([[0], [0], [0.135]])
    rot_3 = rot_y(test3)

    trans2 = np.array([[0], [0], [0.155]])
    rot_2 = rot_y(test2)

    trans1 = np.array([[0.033], [0], [0.019]])
    rot_1 = rot_z(test1)

    # trans0 = np.array([[-0.123], [0], [0.051]])
    trans0=np.array([[-0.147],[0],[0.173]])
    rot_0 = rot_z(DegToRad(180))

    object2base = (
        trans0
        + rot_0 @ (
            rot_1 @ (
                trans1 + rot_2 @ (
                    trans2 + rot_3 @ (
                        trans3 + rot_4 @ (
                            trans4 + rot_5 @ (
                                rot_camera @ (
                                    trans5 + from_eef_to_object
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return object2base

def get_qr_pose_in_camera(points, color_intr):
    """
    QRの4隅 points から、camera座標系におけるQR姿勢を推定する。
    return:
        R_camera_qr: camera座標系から見たQR回転 3x3
        t_camera_qr: camera座標系から見たQR中心位置 3x1
    """

    half = QR_SIZE_M / 2.0

    # QRローカル座標系上の4隅
    # OpenCV QRCodeDetector の points は基本的に
    # 左上, 右上, 右下, 左下 の順になることが多い
    object_points = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    image_points = points.astype(np.float32)

    camera_matrix = np.array([
        [color_intr.fx, 0.0, color_intr.ppx],
        [0.0, color_intr.fy, color_intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dist_coeffs = np.array(color_intr.coeffs, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    R_camera_qr, _ = cv2.Rodrigues(rvec)

    return R_camera_qr, tvec, rvec

def draw_qr_axes(image, rvec, tvec, color_intr, axis_length=0.1):
    """
    QRローカル座標系のXYZ軸を画像上に描画する。
    X: 赤
    Y: 緑
    Z: 青
    """

    camera_matrix = np.array([
        [color_intr.fx, 0.0, color_intr.ppx],
        [0.0, color_intr.fy, color_intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dist_coeffs = np.array(color_intr.coeffs, dtype=np.float32)

    # QRローカル座標系での軸
    axis_points = np.array([
        [0.0, 0.0, 0.0],              # 原点
        [axis_length, 0.0, 0.0],      # X軸
        [0.0, axis_length, 0.0],      # Y軸
        [0.0, 0.0, axis_length],      # Z軸
    ], dtype=np.float32)

    image_points, _ = cv2.projectPoints(
        axis_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    image_points = image_points.reshape(-1, 2).astype(int)

    origin = tuple(image_points[0])
    x_axis = tuple(image_points[1])
    y_axis = tuple(image_points[2])
    z_axis = tuple(image_points[3])

    # X軸: 赤
    cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3, tipLength=0.2)
    cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Y軸: 緑
    cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3, tipLength=0.2)
    cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Z軸: 青
    cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3, tipLength=0.2)
    cv2.putText(image, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def camerainfo_callback(msg):
    global color_intr

    color_intr.height = msg.height
    color_intr.width = msg.width
    color_intr.fx = msg.K[0]
    color_intr.fy = msg.K[4]
    color_intr.ppx = msg.K[2]
    color_intr.ppy = msg.K[5]
    color_intr.model = rs.distortion.inverse_brown_conrady
    color_intr.coeffs = msg.D


def position_callback(msg):
    global cameraPosition
    cameraPosition = msg

def GetCameraToBaseRotation():
    """
    現在の関節角から、camera座標系 -> base座標系 の回転を返す
    """

    if len(cameraPosition.positions) < 5:
        rospy.logwarn("JointPositions がまだ取得できていません")
        return None

    test1 = -(cameraPosition.positions[0].value - DegToRad(169))
    test2 =  (cameraPosition.positions[1].value - DegToRad(65))
    test3 =  (cameraPosition.positions[2].value + DegToRad(146))
    test4 =  (cameraPosition.positions[3].value - DegToRad(102.5))
    test5 = -(cameraPosition.positions[4].value - DegToRad(167.5))

    rot_camera = rot_y(DegToRad(-8))
    rot_5 = rot_z(test5)
    rot_4 = rot_y(test4)
    rot_3 = rot_y(test3)
    rot_2 = rot_y(test2)
    rot_1 = rot_z(test1)
    rot_0 = rot_z(DegToRad(180))

    R_base_camera = rot_0 @ rot_1 @ rot_2 @ rot_3 @ rot_4 @ rot_5 @ rot_camera

    return R_base_camera

def image_callback(depth_msg, color_msg):
    global bridge

    try:
        color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # QR検出

    global frame_count, last_points, last_data

    frame_count += 1

    if frame_count % detect_interval == 0:
        search_image = color_image.copy()

        retval, decoded_info, detected_points, _ = qr_detector.detectAndDecodeMulti(search_image)

        rospy.loginfo("QR multi retval=%s decoded=%s points=%s",
                    retval, decoded_info, detected_points is not None)

        found_main = False

        # 1回目：普通にMainを探す
        if retval and detected_points is not None:
            for data, points_i in zip(decoded_info, detected_points):
                data = data.strip()
                rospy.loginfo("QR candidate data=%s", repr(data))

                if data.startswith("Main"):
                    last_points = points_i
                    last_data = data
                    found_main = True
                    break

        # 2回目：Mainが見つからず、Subなど他のQRが見えているなら、それを白で消して再探索
        if not found_main and retval and detected_points is not None:
            masked_image = color_image.copy()

            for data, points_i in zip(decoded_info, detected_points):
                data = data.strip()

                # Main以外のQR領域を白塗り
                if not data.startswith("Main"):
                    pts = points_i.astype(np.int32)
                    cv2.fillConvexPoly(masked_image, pts, (255, 255, 255))

            retval2, decoded_info2, detected_points2, _ = qr_detector.detectAndDecodeMulti(masked_image)

            rospy.loginfo("QR retry retval=%s decoded=%s points=%s",
                        retval2, decoded_info2, detected_points2 is not None)

            if retval2 and detected_points2 is not None:
                for data, points_i in zip(decoded_info2, detected_points2):
                    data = data.strip()
                    rospy.loginfo("QR retry candidate data=%s", repr(data))

                    if data.startswith("Main"):
                        last_points = points_i
                        last_data = data
                        found_main = True
                        break

        # Mainが見つからなかったら処理しない
        if not found_main:
            last_points = None
            last_data = ""

    # 検出しないフレームでは、前回検出したQR位置を使う
    if last_points is None:
        if frame_count % 10 == 0:
            cv2.imshow("qr_tracking", color_image)
            cv2.waitKey(1)
        return

    points = last_points
    data = last_data


    # QRの4隅から中心を計算
    cx = int(np.mean(points[:, 0]))
    cy = int(np.mean(points[:, 1]))

    # 範囲チェック
    h, w = depth_image.shape[:2]
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return

    z = depth_image[cy, cx]

    # depthがmmかmかを確認
    # 既存コードでは zDepth / 1000 しているので、depth_imageがmm想定。
    # ただしRealSense ROSの32FC1はmの場合もある。
    if z <= 0 or np.isnan(z):
        rospy.logwarn("Invalid depth at QR center")
        return

    # もしzが明らかに大きいなら mm とみなして m に変換
    if z > 10.0:
        z = z / 1000.0

    qr_base = CoordinateTransformation(cx, cy, z, color_intr)

    if qr_base is None:
        return



    R_camera_qr, t_camera_qr, rvec_qr = get_qr_pose_in_camera(points, color_intr)

    if R_camera_qr is None:
        rospy.logwarn("solvePnP failed")
        return

    R_base_object = GetCameraToBaseRotation()

    if R_base_object is None:
        return

    # OpenCV/RealSense camera座標 → 既存の from_eef_to_object と同じ座標系
    R_object_camera = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    R_base_qr = R_base_object @ R_object_camera @ R_camera_qr

    x_base = float(qr_base[0][0])
    y_base = float(qr_base[1][0])
    z_base = float(qr_base[2][0])

    p_base_qr = np.array([
        [x_base],
        [y_base],
        [z_base]
    ])

    # QR座標系から見たBase位置
    p_qr_base = R_base_qr.T @ (-p_base_qr)

    x_qr_base = float(p_qr_base[0][0])
    y_qr_base = float(p_qr_base[1][0])
    z_qr_base = float(p_qr_base[2][0])
    

    pose_msg = PoseStamped()
    pose_msg.header.stamp = color_msg.header.stamp
    pose_msg.header.frame_id = "base_footprint"

    pose_msg.pose.position.x = x_qr_base
    pose_msg.pose.position.y = y_qr_base
    pose_msg.pose.position.z = z_qr_base
    # QR座標系から見たBase姿勢
    # base座標系から見たQR姿勢が R_base_qr なので、
    # その逆回転が QR座標系から見たBase姿勢になる
    R_qr_base = R_base_qr.T

    T_qr_base = np.eye(4)
    T_qr_base[:3, :3] = R_qr_base

    q_qr_base = tft.quaternion_from_matrix(T_qr_base)

    pose_msg.pose.orientation.x = q_qr_base[0]
    pose_msg.pose.orientation.y = q_qr_base[1]
    pose_msg.pose.orientation.z = q_qr_base[2]
    pose_msg.pose.orientation.w = q_qr_base[3]

    qr_pub.publish(pose_msg)

    rospy.loginfo(
        "QR data=%s pixel=(%d,%d) camera_depth=%.3f p_base_qr=(%.3f, %.3f, %.3f) p_qr_base=(%.3f, %.3f, %.3f)",
        data, cx, cy, z,
        x_base, y_base, z_base,
        x_qr_base, y_qr_base, z_qr_base
    )

    # 可視化
    # 可視化
    pts = points.astype(int)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(color_image, p1, p2, (0, 255, 0), 2)

    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

    # QRの姿勢XYZ軸を描画
    draw_qr_axes(color_image, rvec_qr, t_camera_qr, color_intr, axis_length=0.1)

    # cv2.imshow("qr_tracking", color_image)
    # cv2.waitKey(1)
    if frame_count % 10 == 0:
        cv2.imshow("qr_tracking", color_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("qr_tracking_from_realsense")

    qr_pub = rospy.Publisher("/QRposition", PoseStamped, queue_size=10)

    depth_sub = message_filters.Subscriber(
        "/camera/aligned_depth_to_color/image_raw",
        Image
    )
    color_sub = message_filters.Subscriber(
        "/camera/color/image_raw",
        Image
    )

    sync = message_filters.ApproximateTimeSynchronizer(
        [depth_sub, color_sub],
        queue_size=10,
        slop=0.03
    )
    sync.registerCallback(image_callback)

    rospy.Subscriber(
        "/camera/color/camera_info",
        CameraInfo,
        camerainfo_callback
    )

    rospy.Subscriber(
        "arm_2/arm_controller/position_command",
        JointPositions,
        position_callback
    )

    rospy.loginfo("QR Tracking node started")

    try:
        rospy.spin()
    finally:
        cv2.destroyAllWindows()