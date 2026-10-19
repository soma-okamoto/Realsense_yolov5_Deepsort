#!/usr/bin/env python3
"""Meta Quest color/depth overlay viewer.

This node is intentionally independent from the RealSense processing nodes.
It subscribes to the Meta Quest topics only and never subscribes to /tf.
YOLO inference is optional and is applied only to the Meta Quest Color image.
"""

from collections import deque
from pathlib import Path
import sys
import threading

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage


FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "yolov5") not in sys.path:
    sys.path.append(str(ROOT / "yolov5"))


COLOR_TOPIC = "/meta_color_image"
DEPTH_TOPIC = "/meta_depth_image"
COLOR_INFO_TOPIC = "/meta_color_camera_info"
DEPTH_INFO_TOPIC = "/meta_depth_camera_info"
META_TF_TOPIC = "/meta_tf"
META_COLOR_POSE_QR_TOPIC = "/meta_color_pose_qr"
QRP_POSITION_TOPIC = "/QRposition"
META_OBJECT_POSITION_BASE_TOPIC = "/meta_object_position_base"
P_YOLO_TOPIC = "/P_yolo"

COLOR_FRAME = "quest_color_left_optical"
DEPTH_FRAME = "quest_depth_left_optical"
QR_FRAME = "qr_main"
ROBOT_BASE_FRAME = "robot_base"


def stamp_to_seconds(stamp):
    """Return a ROS time as seconds without depending on tf2 utilities."""
    return stamp.secs + stamp.nsecs * 1e-9


def quaternion_to_rotation_matrix(quaternion):
    """Convert geometry_msgs/Quaternion to a 3x3 rotation matrix."""
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("TF quaternion has zero length")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class MetaOverlayNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.color_info = None
        self.depth_info = None
        self.latest_tf = None
        self.qr_from_base = None
        self.base_from_qr = None
        self.qr_position_frame_id = None
        self.qr_position_stamp = None
        self.qr_position_logged = False
        self.meta_pose_logged = False

        self.logged_color_signature = None
        self.logged_depth_signature = None
        self.display_lock = threading.Lock()
        self.latest_display = None
        self.display_enabled = rospy.get_param("~display", True)
        self.invert_meta_tf = rospy.get_param("~invert_meta_tf", False)
        self.tf_history_size = rospy.get_param("~tf_history_size", 200)
        self.tf_history = deque(maxlen=max(1, int(self.tf_history_size)))

        self.yolo_enabled = rospy.get_param("~enable_yolo", True)
        self.yolo_model = None
        self.yolo_device = None
        self.yolo_names = {}
        self.yolo_imgsz = (640, 640)
        self.yolo_conf_thres = float(rospy.get_param("~yolo_conf_thres", 0.25))
        self.yolo_iou_thres = float(rospy.get_param("~yolo_iou_thres", 0.45))
        self.yolo_max_det = int(rospy.get_param("~yolo_max_det", 100))
        self.yolo_inference_lock = threading.Lock()
        if self.yolo_enabled:
            self._load_yolo()

        self.test_point_enabled = rospy.get_param("~test_point_enabled", False)
        self.test_point_u = int(rospy.get_param("~test_point_u", 320))
        self.test_point_v = int(rospy.get_param("~test_point_v", 240))
        self.publish_p_yolo = rospy.get_param("~publish_p_yolo", True)
        self.object_position_pub = rospy.Publisher(
            META_OBJECT_POSITION_BASE_TOPIC,
            PointStamped,
            queue_size=10,
        )
        self.p_yolo_pub = rospy.Publisher(P_YOLO_TOPIC, Float32MultiArray, queue_size=10)

        # CameraInfo and /meta_tf are deliberately ordinary subscribers.
        # No tf2_ros.TransformListener and no /tf subscription are used here.
        rospy.Subscriber(COLOR_INFO_TOPIC, CameraInfo, self.color_info_callback, queue_size=1)
        rospy.Subscriber(DEPTH_INFO_TOPIC, CameraInfo, self.depth_info_callback, queue_size=1)
        rospy.Subscriber(META_TF_TOPIC, TFMessage, self.meta_tf_callback, queue_size=20)
        rospy.Subscriber(QRP_POSITION_TOPIC, PoseStamped, self.qr_position_callback, queue_size=1)

        color_sub = message_filters.Subscriber(COLOR_TOPIC, Image)
        depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
        color_pose_qr_sub = message_filters.Subscriber(META_COLOR_POSE_QR_TOPIC, PoseStamped)
        slop = rospy.get_param("~sync_slop", 0.10)
        queue_size = rospy.get_param("~sync_queue_size", 10)
        self.image_sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, color_pose_qr_sub],
            queue_size=int(queue_size),
            slop=float(slop),
            allow_headerless=False,
        )
        self.image_sync.registerCallback(self.image_callback)

        if self.display_enabled:
            try:
                cv2.namedWindow("Meta Quest Depth Overlay", cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                self.display_enabled = False
                rospy.logwarn("OpenCV window is unavailable: %s", exc)

        rospy.loginfo("Meta Quest overlay node started")
        rospy.loginfo("Color topic: %s", COLOR_TOPIC)
        rospy.loginfo("Depth topic: %s", DEPTH_TOPIC)
        rospy.loginfo("ColorInfo topic: %s", COLOR_INFO_TOPIC)
        rospy.loginfo("DepthInfo topic: %s", DEPTH_INFO_TOPIC)
        rospy.loginfo("Private TF topic: %s (standard /tf is not used)", META_TF_TOPIC)
        rospy.loginfo("Color pose relative to QR topic: %s", META_COLOR_POSE_QR_TOPIC)
        rospy.loginfo("Cached robot-to-QR pose topic: %s", QRP_POSITION_TOPIC)
        rospy.loginfo("Output frame: %s", ROBOT_BASE_FRAME)
        rospy.loginfo("QR detection is not performed in this node")
        rospy.loginfo("TF direction: depth(%s) -> color(%s), invert=%s", DEPTH_FRAME, COLOR_FRAME, self.invert_meta_tf)
        if self.test_point_enabled:
            rospy.loginfo(
                "Known-point verification enabled: depth pixel=(%d, %d)",
                self.test_point_u,
                self.test_point_v,
            )
        if not self.publish_p_yolo:
            rospy.loginfo("/P_yolo publishing disabled by ~publish_p_yolo")

        self._check_topic_presence()

    def _check_topic_presence(self):
        """Report required pose topics without substituting another world frame."""
        required = (
            (META_COLOR_POSE_QR_TOPIC, "geometry_msgs/PoseStamped"),
            (QRP_POSITION_TOPIC, "geometry_msgs/PoseStamped"),
        )
        try:
            published = dict(rospy.get_published_topics())
        except Exception as exc:
            rospy.logwarn("Could not inspect ROS topic types: %s", exc)
            return

        for topic, expected_type in required:
            actual_type = published.get(topic)
            if actual_type is None:
                rospy.logerr(
                    "%s is not currently published. The node will not use Meta world coordinates; "
                    "the placed QRPosition-relative color pose is required.",
                    topic,
                )
            else:
                rospy.loginfo("Topic check: %s type=%s (expected %s)", topic, actual_type, expected_type)

    @staticmethod
    def _pose_stamped_to_matrix(pose_msg):
        """Convert a PoseStamped pose to the transform implied by its header frame."""
        rotation = quaternion_to_rotation_matrix(pose_msg.pose.orientation)
        translation = np.array(
            [
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    @staticmethod
    def _transform_point(transform, point):
        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = np.asarray(point, dtype=np.float64)
        return (transform.dot(point_h))[:3]

    def qr_position_callback(self, msg):
        """Cache /QRposition once as T_qr_from_base; ignore later messages."""
        with self.lock:
            if self.qr_from_base is not None:
                return

        try:
            qr_from_base = self._pose_stamped_to_matrix(msg)
            base_from_qr = np.linalg.inv(qr_from_base)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rospy.logerr("Invalid initial /QRposition; it was not cached: %s", exc)
            return

        with self.lock:
            if self.qr_from_base is not None:
                return
            self.qr_from_base = qr_from_base
            self.base_from_qr = base_from_qr
            self.qr_position_frame_id = msg.header.frame_id
            self.qr_position_stamp = msg.header.stamp

        if msg.header.frame_id != ROBOT_BASE_FRAME:
            rospy.logwarn(
                "/QRposition frame_id=%s, expected %s. Per the requested contract it is treated as "
                "T_qr_from_base without an additional frame conversion.",
                msg.header.frame_id or "<empty>",
                ROBOT_BASE_FRAME,
            )

        # PoseStamped has header.frame_id but no child_frame_id. Do not invent
        # a child frame or infer an extra transform from the message.
        rospy.loginfo(
            "Cached initial /QRposition as T_qr_from_base: frame_id=%s "
            "position=[%.6f, %.6f, %.6f] orientation_xyzw=[%.6f, %.6f, %.6f, %.6f] "
            "(PoseStamped has no child_frame_id)",
            msg.header.frame_id or "<empty>",
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

    def _get_base_from_qr(self):
        with self.lock:
            if self.base_from_qr is None:
                return None
            return self.base_from_qr.copy()

    def _log_color_pose_qr_message(self, color_pose_qr_msg):
        if self.meta_pose_logged:
            return
        self.meta_pose_logged = True
        rospy.loginfo(
            "Using /meta_color_pose_qr as T_qr_from_color: frame_id=%s "
            "stamp=%.6f position=[%.6f, %.6f, %.6f] "
            "orientation_xyzw=[%.6f, %.6f, %.6f, %.6f]",
            color_pose_qr_msg.header.frame_id or "<empty>",
            stamp_to_seconds(color_pose_qr_msg.header.stamp),
            color_pose_qr_msg.pose.position.x,
            color_pose_qr_msg.pose.position.y,
            color_pose_qr_msg.pose.position.z,
            color_pose_qr_msg.pose.orientation.x,
            color_pose_qr_msg.pose.orientation.y,
            color_pose_qr_msg.pose.orientation.z,
            color_pose_qr_msg.pose.orientation.w,
        )

    def _color_pose_to_base(self, color_point, color_pose_qr_msg):
        """Apply T_qr_from_color and inverse(T_qr_from_base)."""
        try:
            qr_from_color = self._pose_stamped_to_matrix(color_pose_qr_msg)
        except ValueError as exc:
            rospy.logwarn_throttle(2.0, "Invalid /meta_color_pose_qr orientation: %s", exc)
            return None

        if color_pose_qr_msg.header.frame_id != QR_FRAME:
            rospy.logwarn_throttle(
                5.0,
                "/meta_color_pose_qr frame_id=%s, expected %s; using the message as T_qr_from_color",
                color_pose_qr_msg.header.frame_id or "<empty>",
                QR_FRAME,
            )

        base_from_qr = self._get_base_from_qr()
        if base_from_qr is None:
            return None

        point_qr = self._transform_point(qr_from_color, color_point)
        return self._transform_point(base_from_qr, point_qr)

    def _depth_to_color_point(self, point_depth, tf_record):
        """Apply p_color = T_color_from_depth * p_depth using /meta_tf."""
        rotation = tf_record["rotation"]
        translation = tf_record["translation"]
        if self.invert_meta_tf:
            rotation = rotation.T
            translation = -rotation.dot(translation)
        return rotation.dot(np.asarray(point_depth, dtype=np.float64)) + translation

    def _publish_base_point(self, point_base, stamp):
        message = PointStamped()
        message.header.stamp = stamp
        message.header.frame_id = ROBOT_BASE_FRAME
        message.point.x = float(point_base[0])
        message.point.y = float(point_base[1])
        message.point.z = float(point_base[2])
        self.object_position_pub.publish(message)

    def _publish_p_yolo(self, bottle_positions_base):
        """Publish existing /P_yolo Float32MultiArray records (12 floats/object)."""
        if not self.publish_p_yolo:
            return

        covariance = np.diag([0.005 ** 2, 0.005 ** 2, 0.010 ** 2])
        message = Float32MultiArray()
        for point_base in bottle_positions_base:
            # Keep the existing track_P_yolo record layout:
            # mean XYZ (3) + row-major 3x3 covariance (9).
            message.data.extend(
                np.asarray(point_base, dtype=np.float32).tolist()
                + covariance.astype(np.float32).reshape(-1).tolist()
            )
        self.p_yolo_pub.publish(message)

    def _verify_known_depth_point(self, depth, depth_info, tf_record, color_pose_qr_msg, stamp):
        """Validate the requested depth->color->QR->base chain with one known pixel."""
        u = self.test_point_u
        v = self.test_point_v
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            rospy.logwarn_throttle(
                2.0,
                "Known test pixel (%d, %d) is outside depth image %s",
                u,
                v,
                depth.shape,
            )
            return

        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0.0:
            rospy.logwarn_throttle(2.0, "Known test pixel (%d, %d) has invalid depth=%s", u, v, z)
            return

        kd = depth_info["K"]
        point_depth = np.array(
            [
                (u - kd[0, 2]) * z / kd[0, 0],
                (v - kd[1, 2]) * z / kd[1, 1],
                z,
            ],
            dtype=np.float64,
        )
        point_color = self._depth_to_color_point(point_depth, tf_record)
        point_base = self._color_pose_to_base(point_color, color_pose_qr_msg)
        if point_base is None:
            rospy.logwarn_throttle(2.0, "Known-point verification is waiting for initial /QRposition")
            return

        self._publish_base_point(point_base, stamp)
        rospy.loginfo_throttle(
            2.0,
            "Known-point chain depth=(%.4f, %.4f, %.4f) color=(%.4f, %.4f, %.4f) "
            "base=(%.4f, %.4f, %.4f) frame_id=%s",
            point_depth[0],
            point_depth[1],
            point_depth[2],
            point_color[0],
            point_color[1],
            point_color[2],
            point_base[0],
            point_base[1],
            point_base[2],
            ROBOT_BASE_FRAME,
        )

    def _load_yolo(self):
        """Load the local YOLOv5 model without importing RealSense or robot code."""
        try:
            import torch
            from yolov5.models.common import DetectMultiBackend
            from yolov5.utils.general import check_img_size
            from yolov5.utils.torch_utils import select_device

            self._torch = torch
            self._check_img_size = check_img_size
            self._select_device = select_device

            weights = rospy.get_param("~yolo_weights", str(ROOT / "best2.pt"))
            data = rospy.get_param("~yolo_data", str(ROOT / "data" / "data.yaml"))
            device_name = rospy.get_param("~yolo_device", "")
            use_half = bool(rospy.get_param("~yolo_half", False))
            requested_imgsz = int(rospy.get_param("~yolo_img_size", 640))

            self.yolo_device = select_device(device_name)
            self.yolo_model = DetectMultiBackend(
                weights,
                device=self.yolo_device,
                dnn=False,
                data=data,
                fp16=use_half,
            )
            self.yolo_imgsz = check_img_size(
                (requested_imgsz, requested_imgsz),
                s=self.yolo_model.stride,
            )
            self.yolo_names = self.yolo_model.names
            self.yolo_model.warmup(imgsz=(1, 3, *self.yolo_imgsz))
            rospy.loginfo(
                "YOLO enabled: weights=%s device=%s imgsz=%s conf=%.2f iou=%.2f",
                weights,
                self.yolo_device,
                self.yolo_imgsz,
                self.yolo_conf_thres,
                self.yolo_iou_thres,
            )
        except Exception as exc:
            # Overlay verification remains usable even if the optional YOLO
            # dependencies or model are unavailable on the ROS machine.
            self.yolo_model = None
            self.yolo_enabled = False
            rospy.logerr("YOLO initialization failed; continuing without YOLO: %s", exc)

    @staticmethod
    def _camera_info_snapshot(msg):
        return {
            "stamp": msg.header.stamp,
            "frame_id": msg.header.frame_id,
            "height": int(msg.height),
            "width": int(msg.width),
            "K": np.asarray(msg.K, dtype=np.float64).reshape(3, 3).copy(),
            "D": np.asarray(msg.D, dtype=np.float64).copy(),
        }

    def color_info_callback(self, msg):
        snapshot = self._camera_info_snapshot(msg)
        with self.lock:
            changed = self.color_info is None or not np.array_equal(self.color_info["K"], snapshot["K"])
            self.color_info = snapshot
        if changed:
            rospy.loginfo(
                "Color CameraInfo: size=%dx%d frame_id=%s K=%s",
                snapshot["width"],
                snapshot["height"],
                snapshot["frame_id"] or "<empty>",
                np.array2string(snapshot["K"], precision=12, separator=", "),
            )

    def depth_info_callback(self, msg):
        snapshot = self._camera_info_snapshot(msg)
        with self.lock:
            changed = self.depth_info is None or not np.array_equal(self.depth_info["K"], snapshot["K"])
            self.depth_info = snapshot
        if changed:
            rospy.loginfo(
                "Depth CameraInfo: size=%dx%d frame_id=%s K=%s",
                snapshot["width"],
                snapshot["height"],
                snapshot["frame_id"] or "<empty>",
                np.array2string(snapshot["K"], precision=12, separator=", "),
            )

    def meta_tf_callback(self, msg):
        """Keep only the private Quest color/depth transform from /meta_tf."""
        for transform_stamped in msg.transforms:
            parent = transform_stamped.header.frame_id
            child = transform_stamped.child_frame_id
            if parent != COLOR_FRAME or child != DEPTH_FRAME:
                continue

            try:
                rotation = quaternion_to_rotation_matrix(transform_stamped.transform.rotation)
            except ValueError as exc:
                rospy.logwarn_throttle(2.0, "Ignoring /meta_tf transform: %s", exc)
                continue

            translation = np.array(
                [
                    transform_stamped.transform.translation.x,
                    transform_stamped.transform.translation.y,
                    transform_stamped.transform.translation.z,
                ],
                dtype=np.float64,
            )
            stamp = transform_stamped.header.stamp
            # A zero TF stamp is not useful for image-time matching. Keep it,
            # but use receipt time as its matching timestamp.
            matching_stamp = stamp if not stamp.is_zero() else rospy.Time.now()
            record = {
                "stamp": matching_stamp,
                "source_stamp": stamp,
                "parent": parent,
                "child": child,
                "translation": translation,
                "rotation": rotation,
            }
            with self.lock:
                self.tf_history.append(record)
                self.latest_tf = record

            q = transform_stamped.transform.rotation
            rospy.loginfo(
                "/meta_tf parent=%s child=%s translation=[%.6f, %.6f, %.6f] "
                "rotation_xyzw=[%.6f, %.6f, %.6f, %.6f]",
                parent,
                child,
                translation[0],
                translation[1],
                translation[2],
                q.x,
                q.y,
                q.z,
                q.w,
            )

    def _select_tf(self, image_stamp):
        with self.lock:
            if not self.tf_history:
                return None, None
            image_time = stamp_to_seconds(image_stamp)
            selected = min(
                self.tf_history,
                key=lambda record: abs(stamp_to_seconds(record["stamp"]) - image_time),
            )
            return selected, abs(stamp_to_seconds(selected["stamp"]) - image_time)

    def _get_camera_infos(self):
        with self.lock:
            if self.color_info is None or self.depth_info is None:
                return None, None
            return self.color_info.copy(), self.depth_info.copy()

    def _convert_color(self, msg):
        if msg.encoding != "rgb8":
            rospy.logwarn_throttle(
                5.0,
                "Color encoding is %s, expected rgb8; attempting conversion",
                msg.encoding,
            )
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Color image is not an HxWx3 image")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _convert_depth(self, msg):
        if msg.encoding != "32FC1":
            rospy.logwarn_throttle(
                5.0,
                "Depth encoding is %s, expected 32FC1; attempting conversion",
                msg.encoding,
            )
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("Depth image is not a single-channel image")
        return depth

    def _project_depth(self, depth, depth_info, color_info, tf_record):
        """Project depth pixels into the color image and apply a Z-buffer."""
        height, width = depth.shape
        kd = depth_info["K"]
        kc = color_info["K"]
        fx_d, fy_d = kd[0, 0], kd[1, 1]
        cx_d, cy_d = kd[0, 2], kd[1, 2]
        fx_c, fy_c = kc[0, 0], kc[1, 1]
        cx_c, cy_c = kc[0, 2], kc[1, 2]
        color_height = int(color_info["height"])
        color_width = int(color_info["width"])
        point_buffer_color = np.zeros((color_height, color_width, 3), dtype=np.float64)

        if fx_d == 0.0 or fy_d == 0.0 or fx_c == 0.0 or fy_c == 0.0:
            raise ValueError("CameraInfo contains a zero focal length")

        valid = np.isfinite(depth) & (depth > 0.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            return (
                np.zeros((color_height, color_width), dtype=np.float32),
                point_buffer_color,
                0,
                0,
                valid_count,
            )

        v_d, u_d = np.nonzero(valid)
        z_d = depth[v_d, u_d].astype(np.float64, copy=False)
        x_d = (u_d.astype(np.float64) - cx_d) * z_d / fx_d
        y_d = (v_d.astype(np.float64) - cy_d) * z_d / fy_d
        points_d = np.column_stack((x_d, y_d, z_d))

        rotation = tf_record["rotation"]
        translation = tf_record["translation"]
        if self.invert_meta_tf:
            # The supplied transform is normally color_T_depth. This option
            # supports streams whose publisher provides the opposite direction.
            rotation = rotation.T
            translation = -rotation.dot(translation)

        points_c = points_d.dot(rotation.T) + translation
        z_c = points_c[:, 2]
        valid_z = np.isfinite(z_c) & (z_c > 0.0)
        if not np.any(valid_z):
            return np.zeros((color_height, color_width), dtype=np.float32), point_buffer_color, 0, 0, valid_count

        points_c = points_c[valid_z]
        z_c = z_c[valid_z]
        u_c = np.rint(fx_c * points_c[:, 0] / z_c + cx_c).astype(np.int32)
        v_c = np.rint(fy_c * points_c[:, 1] / z_c + cy_c).astype(np.int32)

        in_bounds = (u_c >= 0) & (u_c < color_width) & (v_c >= 0) & (v_c < color_height)
        if not np.any(in_bounds):
            return np.zeros((color_height, color_width), dtype=np.float32), point_buffer_color, 0, 0, valid_count

        u_c = u_c[in_bounds]
        v_c = v_c[in_bounds]
        z_c = z_c[in_bounds]
        points_c = points_c[in_bounds]
        color_indices = v_c.astype(np.int64) * color_width + u_c.astype(np.int64)
        projected_count = int(color_indices.size)

        # Sorting by camera-space Z and taking the first pixel occurrence is a
        # deterministic nearest-point Z-buffer implementation.
        order = np.argsort(z_c, kind="mergesort")
        sorted_indices = color_indices[order]
        first_for_pixel = np.ones(sorted_indices.shape, dtype=bool)
        if sorted_indices.size > 1:
            first_for_pixel[1:] = sorted_indices[1:] != sorted_indices[:-1]
        chosen = order[first_for_pixel]

        z_buffer = np.zeros((color_height, color_width), dtype=np.float32)
        z_buffer[v_c[chosen], u_c[chosen]] = z_c[chosen].astype(np.float32)
        point_buffer_color[v_c[chosen], u_c[chosen]] = points_c[chosen]
        return z_buffer, point_buffer_color, projected_count, int(chosen.size), valid_count

    @staticmethod
    def _make_overlay(color_bgr, z_buffer):
        mask = z_buffer > 0.0
        overlay = color_bgr.copy()
        if not np.any(mask):
            return overlay

        values = z_buffer[mask]
        near = float(np.min(values))
        far = float(np.max(values))
        if far > near:
            normalized = np.zeros_like(z_buffer, dtype=np.uint8)
            normalized[mask] = np.clip((far - values) / (far - near) * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            normalized = np.zeros_like(z_buffer, dtype=np.uint8)
            normalized[mask] = 128

        depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay[mask] = cv2.addWeighted(color_bgr[mask], 0.60, depth_color[mask], 0.40, 0.0)
        return overlay

    @staticmethod
    def _make_depth_visualization(depth):
        """Create a standalone false-color view of the received meter depth."""
        mask = np.isfinite(depth) & (depth > 0.0)
        visualization = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        if not np.any(mask):
            return visualization

        values = depth[mask]
        near = float(np.min(values))
        far = float(np.max(values))
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if far > near:
            normalized[mask] = np.clip((far - values) / (far - near) * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            normalized[mask] = 128
        visualization = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        visualization[~mask] = (0, 0, 0)
        return visualization

    @staticmethod
    def _make_projection_debug(color_bgr, z_buffer):
        """Draw sampled projected depth pixels on the original Color image."""
        debug = color_bgr.copy()
        ys, xs = np.nonzero(z_buffer > 0.0)
        if xs.size == 0:
            return debug

        sample_count = min(4000, xs.size)
        sample_indices = np.linspace(0, xs.size - 1, sample_count, dtype=np.int64)
        # Green pixels mark where a depth point landed in the Color image.
        debug[ys[sample_indices], xs[sample_indices]] = (0, 255, 0)
        return debug

    @staticmethod
    def _label_and_resize(image, label):
        labeled = image.copy()
        cv2.rectangle(labeled, (0, 0), (labeled.shape[1] - 1, 30), (0, 0, 0), -1)
        cv2.putText(
            labeled,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return cv2.resize(labeled, (labeled.shape[1] // 2, labeled.shape[0] // 2), interpolation=cv2.INTER_AREA)

    def _make_debug_view(
        self,
        color_bgr,
        depth,
        z_buffer,
        valid_count,
        projected_count,
        zbuffer_count,
        yolo_image=None,
        detection_count=0,
    ):
        depth_view = self._make_depth_visualization(depth)
        annotated = yolo_image if yolo_image is not None else color_bgr
        # YOLO results are intentionally shown only on the Color image.
        # Keep the projection and depth-overlay panes free of detection boxes.
        projection_view = self._make_projection_debug(color_bgr, z_buffer)
        overlay_view = self._make_overlay(color_bgr, z_buffer)

        top = np.hstack(
            (
                self._label_and_resize(annotated, "1: Color + YOLO"),
                self._label_and_resize(depth_view, "2: received Depth (m)"),
            )
        )
        bottom = np.hstack(
            (
                self._label_and_resize(projection_view, "3: projected points (green)"),
                self._label_and_resize(overlay_view, "4: depth overlay"),
            )
        )
        debug_view = np.vstack((top, bottom))
        cv2.putText(
            debug_view,
            "valid={} projected={} zbuffer={} | q/ESC: quit".format(
                valid_count, projected_count, zbuffer_count
            ),
            (10, debug_view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_view,
            "YOLO detections={}".format(detection_count),
            (debug_view.shape[1] - 230, debug_view.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return debug_view

    def _class_name(self, class_id):
        if isinstance(self.yolo_names, dict):
            return str(self.yolo_names.get(class_id, class_id))
        if 0 <= class_id < len(self.yolo_names):
            return str(self.yolo_names[class_id])
        return str(class_id)

    @staticmethod
    def _point_at_color_pixel(point_buffer_color, u, v):
        height, width = point_buffer_color.shape[:2]
        if 0 <= u < width and 0 <= v < height:
            point = point_buffer_color[v, u]
            if np.all(np.isfinite(point)) and np.linalg.norm(point) > 0.0:
                return point.copy()

        # Bounding-box centers can fall on a hole in the projected Z-buffer.
        # Use a small valid neighborhood only as a display fallback.
        radius = 4
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        neighborhood = point_buffer_color[y0:y1, x0:x1].reshape(-1, 3)
        valid = neighborhood[np.isfinite(neighborhood).all(axis=1)]
        valid = valid[np.linalg.norm(valid, axis=1) > 0.0]
        return np.median(valid, axis=0) if valid.size else None

    def _annotate_yolo(self, color_bgr, detections, point_buffer_color, color_pose_qr_msg):
        """Draw YOLO boxes and show bottle coordinates in robot_base."""
        annotated = color_bgr.copy()
        detection_count = 0
        bottle_positions_base = []

        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection[:4]]
            confidence = float(detection[4])
            class_id = int(detection[5])
            class_name = self._class_name(class_id)
            center_u = int(round((x1 + x2) * 0.5))
            center_v = int(round((y1 + y2) * 0.5))
            point_color = self._point_at_color_pixel(point_buffer_color, center_u, center_v)

            label = "{} {:.2f}".format(class_name, confidence)
            coordinate_label = None
            if class_name.lower() == "bottle":
                if point_color is None:
                    coordinate_label = "base XYZ=NA(depth)"
                else:
                    point_base = self._color_pose_to_base(point_color, color_pose_qr_msg)
                    if point_base is None:
                        coordinate_label = "base XYZ=WAIT(/QRposition)"
                    else:
                        coordinate_label = "base XYZ=({:.2f}, {:.2f}, {:.2f})m".format(
                            point_base[0], point_base[1], point_base[2]
                        )
                        bottle_positions_base.append(point_base)

            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            self._draw_label(annotated, label, x1, y1, color)
            if coordinate_label is not None:
                self._draw_label(annotated, coordinate_label, x1, y1 + 24, color)
            detection_count += 1

        return annotated, detection_count, bottle_positions_base

    @staticmethod
    def _draw_label(image, text, x, y, color):
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_y = max(y, text_size[1] + baseline + 2)
        x2 = min(image.shape[1] - 1, x + text_size[0] + 4)
        cv2.rectangle(
            image,
            (x, text_y - text_size[1] - baseline - 2),
            (x2, text_y),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def _run_yolo(self, color_bgr, point_buffer_color, color_pose_qr_msg):
        if self.yolo_model is None:
            return color_bgr, 0, []

        try:
            from yolov5.utils.augmentations import letterbox
            from yolov5.utils.general import non_max_suppression, scale_coords

            # Prevent overlapping inference callbacks if the camera rate is
            # higher than the selected YOLO model can process.
            if not self.yolo_inference_lock.acquire(blocking=False):
                return color_bgr, 0, []
            try:
                image, _, _ = letterbox(
                    color_bgr,
                    new_shape=self.yolo_imgsz,
                    stride=self.yolo_model.stride,
                    auto=self.yolo_model.pt,
                )
                image = image[:, :, ::-1].transpose(2, 0, 1)
                image = np.ascontiguousarray(image)
                tensor = self._torch.from_numpy(image).to(self.yolo_device)
                tensor = tensor.half() if self.yolo_model.fp16 else tensor.float()
                tensor /= 255.0
                if tensor.ndimension() == 3:
                    tensor = tensor[None]

                with self._torch.no_grad():
                    prediction = self.yolo_model(tensor, augment=False, visualize=False)
                prediction = non_max_suppression(
                    prediction,
                    self.yolo_conf_thres,
                    self.yolo_iou_thres,
                    classes=None,
                    agnostic=False,
                    max_det=self.yolo_max_det,
                )
                detections = prediction[0]
                if detections is None or not len(detections):
                    return color_bgr, 0, []

                detections[:, :4] = scale_coords(
                    tensor.shape[2:],
                    detections[:, :4],
                    color_bgr.shape,
                ).round()
                return self._annotate_yolo(color_bgr, detections, point_buffer_color, color_pose_qr_msg)
            finally:
                self.yolo_inference_lock.release()
        except Exception as exc:
            rospy.logerr_throttle(2.0, "YOLO inference failed: %s", exc)
            return color_bgr, 0, []

    def image_callback(self, color_msg, depth_msg, color_pose_qr_msg):
        color_signature = (color_msg.encoding, color_msg.width, color_msg.height, color_msg.header.frame_id)
        depth_signature = (depth_msg.encoding, depth_msg.width, depth_msg.height, depth_msg.header.frame_id)
        if color_signature != self.logged_color_signature:
            self.logged_color_signature = color_signature
            rospy.loginfo(
                "Color image: encoding=%s size=%dx%d frame_id=%s",
                color_msg.encoding,
                color_msg.width,
                color_msg.height,
                color_msg.header.frame_id or "<empty>",
            )
        if depth_signature != self.logged_depth_signature:
            self.logged_depth_signature = depth_signature
            rospy.loginfo(
                "Depth image: encoding=%s size=%dx%d frame_id=%s",
                depth_msg.encoding,
                depth_msg.width,
                depth_msg.height,
                depth_msg.header.frame_id or "<empty>",
            )

        color_info, depth_info = self._get_camera_infos()
        if color_info is None or depth_info is None:
            rospy.logwarn_throttle(2.0, "Waiting for both Meta Quest CameraInfo messages")
            return

        tf_record, tf_delta = self._select_tf(color_msg.header.stamp)
        if tf_record is None:
            rospy.logwarn_throttle(2.0, "Waiting for /meta_tf %s -> %s", COLOR_FRAME, DEPTH_FRAME)
            return

        if color_pose_qr_msg.header.frame_id and color_pose_qr_msg.header.frame_id != QR_FRAME:
            rospy.logwarn_throttle(
                5.0,
                "/meta_color_pose_qr frame_id=%s, expected %s",
                color_pose_qr_msg.header.frame_id,
                QR_FRAME,
            )
        self._log_color_pose_qr_message(color_pose_qr_msg)

        try:
            color_bgr = self._convert_color(color_msg)
            depth = self._convert_depth(depth_msg)
            z_buffer, point_buffer_color, projected_count, zbuffer_count, valid_count = self._project_depth(
                depth,
                depth_info,
                color_info,
                tf_record,
            )
        except (CvBridgeError, ValueError, cv2.error) as exc:
            rospy.logerr_throttle(2.0, "Meta Quest image processing failed: %s", exc)
            return

        rospy.loginfo_throttle(
            5.0,
            "Color intensity range=[%d, %d], Depth dtype=%s shape=%s",
            int(np.min(color_bgr)),
            int(np.max(color_bgr)),
            depth.dtype,
            depth.shape,
        )

        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size:
            depth_min = float(np.min(finite_depth))
            depth_max = float(np.max(finite_depth))
        else:
            depth_min = float("nan")
            depth_max = float("nan")

        rospy.loginfo_throttle(
            1.0,
            "Depth range=[%.4f, %.4f] m, valid points=%d, projected points=%d, "
            "Z-buffer pixels=%d, TF image delta=%.4f s",
            depth_min,
            depth_max,
            valid_count,
            projected_count,
            zbuffer_count,
            tf_delta,
        )

        if self.test_point_enabled:
            self._verify_known_depth_point(
                depth,
                depth_info,
                tf_record,
                color_pose_qr_msg,
                color_msg.header.stamp,
            )

        yolo_image, detection_count, bottle_positions_base = self._run_yolo(
            color_bgr,
            point_buffer_color,
            color_pose_qr_msg,
        )
        for point_base in bottle_positions_base:
            self._publish_base_point(point_base, color_msg.header.stamp)
        self._publish_p_yolo(bottle_positions_base)
        rospy.loginfo_throttle(
            2.0,
            "YOLO detections=%d bottles_in_robot_base=%d",
            detection_count,
            len(bottle_positions_base),
        )

        if not self.display_enabled:
            return

        display = self._make_debug_view(
            color_bgr,
            depth,
            z_buffer,
            valid_count,
            projected_count,
            zbuffer_count,
            yolo_image=yolo_image,
            detection_count=detection_count,
        )
        with self.display_lock:
            self.latest_display = display

    def show_latest(self):
        """Update the GUI from the main thread instead of a ROS callback thread."""
        if not self.display_enabled:
            return
        with self.display_lock:
            if self.latest_display is None:
                return
            display = self.latest_display.copy()
        try:
            cv2.imshow("Meta Quest Depth Overlay", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.signal_shutdown("OpenCV window closed by user")
        except cv2.error as exc:
            self.display_enabled = False
            rospy.logwarn("OpenCV display disabled: %s", exc)

    def shutdown(self):
        if self.display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def main():
    rospy.init_node("track_P_meta", anonymous=False)
    node = MetaOverlayNode()
    rospy.on_shutdown(node.shutdown)
    # Keep OpenCV GUI calls in the main thread. This avoids black/non-updating
    # windows on systems where GUI backends do not work from ROS callback threads.
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        node.show_latest()
        rate.sleep()


if __name__ == "__main__":
    main()
