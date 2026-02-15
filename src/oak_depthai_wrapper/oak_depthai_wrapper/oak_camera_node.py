"""
OAK-D Pro camera node using DepthAI.

Publishes RGB, depth-aligned-to-RGB, CameraInfo, and NN segmentation output.
Supports mock NN mode for development without a real .blob model.
"""

import time
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge


class OakCameraNode(Node):
    def __init__(self):
        super().__init__('oak_camera_node')

        # Declare parameters
        self.declare_parameter('resolution_width', 640)
        self.declare_parameter('resolution_height', 480)
        self.declare_parameter('fps', 15)
        self.declare_parameter('ir_projector_enable', False)
        self.declare_parameter('exposure_us', 0)  # 0 = auto
        self.declare_parameter('model_blob_path', '')
        self.declare_parameter('enable_mock_nn', True)
        self.declare_parameter('nn_input_width', 256)
        self.declare_parameter('nn_input_height', 256)
        self.declare_parameter('device_reconnect_interval_s', 5.0)

        self._width = self.get_parameter('resolution_width').value
        self._height = self.get_parameter('resolution_height').value
        self._fps = self.get_parameter('fps').value
        self._ir_enable = self.get_parameter('ir_projector_enable').value
        self._exposure_us = self.get_parameter('exposure_us').value
        self._blob_path = self.get_parameter('model_blob_path').value
        self._mock_nn = self.get_parameter('enable_mock_nn').value
        self._nn_w = self.get_parameter('nn_input_width').value
        self._nn_h = self.get_parameter('nn_input_height').value
        self._reconnect_interval = self.get_parameter('device_reconnect_interval_s').value

        # Publishers
        self._pub_rgb = self.create_publisher(Image, '/oak/rgb/image_raw', 10)
        self._pub_depth = self.create_publisher(Image, '/oak/depth_aligned/image_raw', 10)
        self._pub_info = self.create_publisher(CameraInfo, '/oak/camera_info', 10)
        self._pub_nn = self.create_publisher(Image, '/oak/nn/segmentation_raw', 10)

        self._bridge = CvBridge()
        self._device = None
        self._pipeline = None

        # Try to import depthai
        self._dai = None
        try:
            import depthai as dai
            self._dai = dai
            self.get_logger().info('DepthAI library found.')
        except ImportError:
            self.get_logger().warn(
                'DepthAI library not found. Running in mock-only mode.'
            )
            self._mock_nn = True

        if self._mock_nn:
            self.get_logger().info('Mock NN mode enabled — publishing synthetic data.')
            self._timer = self.create_timer(1.0 / self._fps, self._mock_callback)
        else:
            self._connect_device()

    # ------------------------------------------------------------------ #
    #  REAL DEVICE PATH
    # ------------------------------------------------------------------ #
    def _connect_device(self):
        """Build DepthAI pipeline and connect to OAK device."""
        dai = self._dai
        if dai is None:
            self.get_logger().error('Cannot connect: depthai not installed.')
            return

        try:
            pipeline = dai.Pipeline()

            # --- RGB camera ---
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(self._width, self._height)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam_rgb.setFps(self._fps)
            if self._exposure_us > 0:
                cam_rgb.initialControl.setManualExposure(self._exposure_us, 100)

            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName('rgb')
            cam_rgb.preview.link(xout_rgb.input)

            # --- Mono cameras + stereo depth ---
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_left.setResolution(
                dai.MonoCameraProperties.SensorResolution.THE_400_P
            )
            mono_left.setCamera('left')
            mono_left.setFps(self._fps)

            mono_right = pipeline.create(dai.node.MonoCamera)
            mono_right.setResolution(
                dai.MonoCameraProperties.SensorResolution.THE_400_P
            )
            mono_right.setCamera('right')
            mono_right.setFps(self._fps)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.HIGH_DENSITY
            )
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(self._width, self._height)
            stereo.setSubpixel(True)

            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName('depth')
            stereo.depth.link(xout_depth.input)

            # --- NN (segmentation blob) ---
            if self._blob_path:
                nn = pipeline.create(dai.node.NeuralNetwork)
                nn.setBlobPath(self._blob_path)
                nn.setNumInferenceThreads(2)
                nn.input.setBlocking(False)
                nn.input.setQueueSize(1)

                # Resize RGB for NN input
                manip = pipeline.create(dai.node.ImageManip)
                manip.initialConfig.setResize(self._nn_w, self._nn_h)
                manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
                cam_rgb.preview.link(manip.inputImage)
                manip.out.link(nn.input)

                xout_nn = pipeline.create(dai.node.XLinkOut)
                xout_nn.setStreamName('nn')
                nn.out.link(xout_nn.input)

            self._pipeline = pipeline
            self._device = dai.Device(pipeline)

            # IR projector
            if self._ir_enable:
                self._device.setIrLaserDotProjectorIntensity(0.7)

            self._q_rgb = self._device.getOutputQueue('rgb', maxSize=4, blocking=False)
            self._q_depth = self._device.getOutputQueue('depth', maxSize=4, blocking=False)
            self._q_nn = None
            if self._blob_path:
                self._q_nn = self._device.getOutputQueue('nn', maxSize=4, blocking=False)

            # Get calibration for CameraInfo
            calib = self._device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, self._width, self._height
            )
            self._camera_info_msg = self._build_camera_info(intrinsics)

            self.get_logger().info(
                f'OAK device connected. Resolution {self._width}x{self._height} @ {self._fps}fps'
            )

            self._timer = self.create_timer(1.0 / self._fps, self._device_callback)

        except Exception as e:
            self.get_logger().error(f'Failed to connect to OAK device: {e}')
            self.get_logger().info(
                f'Retrying in {self._reconnect_interval}s ...'
            )
            self._timer = self.create_timer(
                self._reconnect_interval, self._retry_connect
            )

    def _retry_connect(self):
        """Timer callback to retry device connection."""
        self._timer.cancel()
        self.get_logger().info('Retrying OAK device connection...')
        self._connect_device()

    def _build_camera_info(self, intrinsics):
        """Build CameraInfo message from intrinsic matrix."""
        msg = CameraInfo()
        msg.header.frame_id = 'oak_rgb_optical_frame'
        msg.width = self._width
        msg.height = self._height
        msg.distortion_model = 'plumb_bob'
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return msg

    def _device_callback(self):
        """Publish frames from the real OAK device."""
        try:
            stamp = self.get_clock().now().to_msg()

            # RGB
            in_rgb = self._q_rgb.tryGet()
            if in_rgb is not None:
                frame_rgb = in_rgb.getCvFrame()
                rgb_msg = self._bridge.cv2_to_imgmsg(frame_rgb, encoding='bgr8')
                rgb_msg.header.stamp = stamp
                rgb_msg.header.frame_id = 'oak_rgb_optical_frame'
                self._pub_rgb.publish(rgb_msg)

            # Depth
            in_depth = self._q_depth.tryGet()
            if in_depth is not None:
                frame_depth = in_depth.getFrame()  # uint16 mm
                depth_msg = self._bridge.cv2_to_imgmsg(frame_depth, encoding='16UC1')
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = 'oak_depth_optical_frame'
                self._pub_depth.publish(depth_msg)

            # CameraInfo
            self._camera_info_msg.header.stamp = stamp
            self._pub_info.publish(self._camera_info_msg)

            # NN output
            if self._q_nn is not None:
                in_nn = self._q_nn.tryGet()
                if in_nn is not None:
                    # Get first layer output as flat array, reshape to HxW label mask
                    layer = in_nn.getFirstLayerFp16()
                    nn_out = np.array(layer, dtype=np.float32)
                    # Assume output shape is (num_classes, H, W) — take argmax
                    num_classes = 3
                    nn_out = nn_out.reshape(num_classes, self._nn_h, self._nn_w)
                    label_mask = np.argmax(nn_out, axis=0).astype(np.uint8)
                    nn_msg = self._bridge.cv2_to_imgmsg(label_mask, encoding='mono8')
                    nn_msg.header.stamp = stamp
                    nn_msg.header.frame_id = 'oak_rgb_optical_frame'
                    self._pub_nn.publish(nn_msg)

        except Exception as e:
            self.get_logger().error(f'Device read error: {e}')
            self.get_logger().info('Device may have disconnected. Attempting reconnect...')
            self._timer.cancel()
            if self._device is not None:
                try:
                    self._device.close()
                except Exception:
                    pass
                self._device = None
            self._timer = self.create_timer(
                self._reconnect_interval, self._retry_connect
            )

    # ------------------------------------------------------------------ #
    #  MOCK PATH
    # ------------------------------------------------------------------ #
    def _mock_callback(self):
        """Publish synthetic data for development without an OAK device."""
        stamp = self.get_clock().now().to_msg()

        # Synthetic RGB (gradient)
        rgb = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        rgb[:, :, 1] = np.linspace(0, 200, self._width, dtype=np.uint8)
        rgb_msg = self._bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = 'oak_rgb_optical_frame'
        self._pub_rgb.publish(rgb_msg)

        # Synthetic depth (planar)
        depth = np.full((self._height, self._width), 500, dtype=np.uint16)
        depth_msg = self._bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'oak_depth_optical_frame'
        self._pub_depth.publish(depth_msg)

        # Synthetic CameraInfo
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = 'oak_rgb_optical_frame'
        info.width = self._width
        info.height = self._height
        info.distortion_model = 'plumb_bob'
        fx = fy = 500.0
        cx = self._width / 2.0
        cy = self._height / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._pub_info.publish(info)

        # Synthetic NN segmentation (striped pattern)
        nn_mask = np.zeros((self._nn_h, self._nn_w), dtype=np.uint8)
        nn_mask[self._nn_h // 3 : 2 * self._nn_h // 3, :] = 1  # LEAF
        nn_mask[2 * self._nn_h // 3 :, self._nn_w // 3 : 2 * self._nn_w // 3] = 2  # STEM
        nn_msg = self._bridge.cv2_to_imgmsg(nn_mask, encoding='mono8')
        nn_msg.header.stamp = stamp
        nn_msg.header.frame_id = 'oak_rgb_optical_frame'
        self._pub_nn.publish(nn_msg)

    # ------------------------------------------------------------------ #
    def destroy_node(self):
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
