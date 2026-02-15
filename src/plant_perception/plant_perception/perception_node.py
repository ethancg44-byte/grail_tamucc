"""
Plant perception node.

Subscribes to OAK camera outputs and produces:
- Label mask from NN segmentation
- Stem node detection via skeletonization + branch-point analysis
- 3D node positions using depth + intrinsics
- Rule-based cut plan
- Debug overlay image
"""

import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from plant_interfaces.msg import (
    OrganSegmentation,
    StemNode,
    StemNodes,
    CutTarget,
    CutPlan,
)


# Class IDs
BG = 0
LEAF = 1
STEM = 2

# Overlay colors (BGR)
CLASS_COLORS = {
    BG: (0, 0, 0),
    LEAF: (0, 180, 0),
    STEM: (0, 120, 255),
}


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Parameters
        self.declare_parameter('class_map_yaml',
                               '{0: BACKGROUND, 1: LEAF, 2: STEM_OR_PETIOLE}')
        self.declare_parameter('skeleton_method', 'opencv')
        self.declare_parameter('branch_kernel_size', 3)
        self.declare_parameter('node_association_max_dist_px', 30)
        self.declare_parameter('node_max_age_frames', 10)
        self.declare_parameter('depth_min_mm', 100)
        self.declare_parameter('depth_max_mm', 3000)
        self.declare_parameter('cut_y_threshold_px', 300)
        self.declare_parameter('cut_leaf_density_radius_px', 50)
        self.declare_parameter('cut_leaf_density_threshold', 0.4)
        self.declare_parameter('cut_min_confidence', 0.3)
        self.declare_parameter('publish_debug_overlay', True)

        self._class_map_yaml = self.get_parameter('class_map_yaml').value
        self._skel_method = self.get_parameter('skeleton_method').value
        self._branch_ks = self.get_parameter('branch_kernel_size').value
        self._assoc_max_dist = self.get_parameter('node_association_max_dist_px').value
        self._node_max_age = self.get_parameter('node_max_age_frames').value
        self._depth_min = self.get_parameter('depth_min_mm').value
        self._depth_max = self.get_parameter('depth_max_mm').value
        self._cut_y_thresh = self.get_parameter('cut_y_threshold_px').value
        self._cut_density_r = self.get_parameter('cut_leaf_density_radius_px').value
        self._cut_density_thresh = self.get_parameter('cut_leaf_density_threshold').value
        self._cut_min_conf = self.get_parameter('cut_min_confidence').value
        self._pub_debug = self.get_parameter('publish_debug_overlay').value

        # Check for OpenCV ximgproc
        self._has_ximgproc = hasattr(cv2, 'ximgproc')
        if self._skel_method == 'opencv' and not self._has_ximgproc:
            self.get_logger().warn(
                'cv2.ximgproc not available; falling back to Zhang-Suen skeletonization.'
            )
            self._skel_method = 'zhang_suen'

        # Publishers
        self._pub_label = self.create_publisher(Image, '/perception/label_mask', 10)
        self._pub_overlay = self.create_publisher(Image, '/debug/overlay_image', 10)
        self._pub_seg = self.create_publisher(
            OrganSegmentation, '/perception/segmentation', 10
        )
        self._pub_nodes = self.create_publisher(StemNodes, '/perception/nodes', 10)
        self._pub_cut = self.create_publisher(CutPlan, '/perception/cut_plan', 10)
        self._pub_markers = self.create_publisher(
            MarkerArray, '/perception/node_markers', 10
        )

        # Subscribers
        self._sub_rgb = self.create_subscription(
            Image, '/oak/rgb/image_raw', self._cb_rgb, 10
        )
        self._sub_depth = self.create_subscription(
            Image, '/oak/depth_aligned/image_raw', self._cb_depth, 10
        )
        self._sub_info = self.create_subscription(
            CameraInfo, '/oak/camera_info', self._cb_info, 10
        )
        self._sub_nn = self.create_subscription(
            Image, '/oak/nn/segmentation_raw', self._cb_nn, 10
        )

        self._bridge = CvBridge()

        # Cached data
        self._rgb = None
        self._depth = None
        self._intrinsics = None  # (fx, fy, cx, cy)
        self._nn_mask = None

        # Temporal node tracking
        self._tracked_nodes = []  # list of dicts: {id, u, v, age}
        self._next_node_id = 0

        self.get_logger().info('Perception node started.')

    # ------------------------------------------------------------------ #
    #  Subscriber callbacks — cache latest data
    # ------------------------------------------------------------------ #
    def _cb_rgb(self, msg: Image):
        self._rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _cb_depth(self, msg: Image):
        self._depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

    def _cb_info(self, msg: CameraInfo):
        k = msg.k
        self._intrinsics = (k[0], k[4], k[2], k[5])  # fx, fy, cx, cy

    def _cb_nn(self, msg: Image):
        """NN segmentation received — run full perception pipeline."""
        self._nn_mask = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self._run_pipeline(msg.header)

    # ------------------------------------------------------------------ #
    #  Main pipeline
    # ------------------------------------------------------------------ #
    def _run_pipeline(self, header):
        if self._nn_mask is None:
            return

        stamp = header.stamp
        label_mask = self._nn_mask.copy()

        # Resize label mask to match RGB if needed
        if self._rgb is not None and label_mask.shape[:2] != self._rgb.shape[:2]:
            label_mask = cv2.resize(
                label_mask,
                (self._rgb.shape[1], self._rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # 1. Publish label mask
        label_msg = self._bridge.cv2_to_imgmsg(label_mask, encoding='mono8')
        label_msg.header.stamp = stamp
        label_msg.header.frame_id = 'oak_rgb_optical_frame'
        self._pub_label.publish(label_msg)

        # 2. Publish OrganSegmentation
        seg_msg = OrganSegmentation()
        seg_msg.header.stamp = stamp
        seg_msg.header.frame_id = 'oak_rgb_optical_frame'
        seg_msg.mask_topic = '/perception/label_mask'
        seg_msg.class_map_yaml = self._class_map_yaml
        self._pub_seg.publish(seg_msg)

        # 3. Skeletonize STEM class and find branch points
        stem_mask = (label_mask == STEM).astype(np.uint8) * 255
        skeleton = self._skeletonize(stem_mask)
        branch_pts = self._find_branch_points(skeleton)

        # 4. Temporal stabilization
        self._update_tracked_nodes(branch_pts)

        # 5. Build StemNodes with 3D projection
        nodes_msg = StemNodes()
        nodes_msg.header.stamp = stamp
        nodes_msg.header.frame_id = 'oak_rgb_optical_frame'

        for tn in self._tracked_nodes:
            node = StemNode()
            node.id = tn['id']
            node.u = tn['u']
            node.v = tn['v']
            node.confidence = max(0.0, 1.0 - tn['age'] / max(self._node_max_age, 1))

            pt = self._project_3d(tn['u'], tn['v'], stamp)
            if pt is not None:
                node.point_camera = pt
            else:
                node.point_camera = PointStamped()
                node.point_camera.header.stamp = stamp
                node.point_camera.header.frame_id = 'oak_rgb_optical_frame'

            nodes_msg.nodes.append(node)

        self._pub_nodes.publish(nodes_msg)

        # 6. Rule-based cut selection
        cut_msg = self._select_cuts(nodes_msg, label_mask, stamp)
        self._pub_cut.publish(cut_msg)

        # 7. Publish markers for RViz
        self._publish_markers(nodes_msg, cut_msg, stamp)

        # 8. Debug overlay
        if self._pub_debug and self._rgb is not None:
            overlay = self._make_overlay(label_mask, skeleton, branch_pts, cut_msg)
            overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header.stamp = stamp
            overlay_msg.header.frame_id = 'oak_rgb_optical_frame'
            self._pub_overlay.publish(overlay_msg)

    # ------------------------------------------------------------------ #
    #  Skeletonization
    # ------------------------------------------------------------------ #
    def _skeletonize(self, binary_mask):
        if self._skel_method == 'opencv' and self._has_ximgproc:
            return cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        else:
            return self._zhang_suen(binary_mask)

    @staticmethod
    def _zhang_suen(image):
        """Zhang-Suen thinning algorithm (pure Python/NumPy fallback)."""
        img = (image > 0).astype(np.uint8)
        changed = True
        while changed:
            changed = False
            for step in range(2):
                marker = np.zeros_like(img)
                rows, cols = img.shape
                for i in range(1, rows - 1):
                    for j in range(1, cols - 1):
                        if img[i, j] != 1:
                            continue
                        p2 = img[i - 1, j]
                        p3 = img[i - 1, j + 1]
                        p4 = img[i, j + 1]
                        p5 = img[i + 1, j + 1]
                        p6 = img[i + 1, j]
                        p7 = img[i + 1, j - 1]
                        p8 = img[i, j - 1]
                        p9 = img[i - 1, j - 1]

                        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                        b = sum(neighbors)
                        if b < 2 or b > 6:
                            continue

                        # Count 0->1 transitions in ordered sequence
                        seq = neighbors + [neighbors[0]]
                        a = sum(
                            1 for k in range(len(seq) - 1)
                            if seq[k] == 0 and seq[k + 1] == 1
                        )
                        if a != 1:
                            continue

                        if step == 0:
                            if p2 * p4 * p6 != 0:
                                continue
                            if p4 * p6 * p8 != 0:
                                continue
                        else:
                            if p2 * p4 * p8 != 0:
                                continue
                            if p2 * p6 * p8 != 0:
                                continue

                        marker[i, j] = 1
                        changed = True

                img[marker == 1] = 0

        return img * 255

    # ------------------------------------------------------------------ #
    #  Branch-point detection
    # ------------------------------------------------------------------ #
    def _find_branch_points(self, skeleton):
        """Find branch points (pixels with >=3 skeleton neighbors)."""
        skel_bin = (skeleton > 0).astype(np.uint8)
        if skel_bin.sum() == 0:
            return []

        # Convolve with 3x3 kernel to count neighbors
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(skel_bin, -1, kernel)

        # Branch points: skeleton pixels with >= 3 neighbors
        bp_mask = (skel_bin > 0) & (neighbor_count >= 3)
        coords = np.argwhere(bp_mask)  # (row, col) = (v, u)

        # Cluster nearby branch points
        if len(coords) == 0:
            return []

        points = []
        used = np.zeros(len(coords), dtype=bool)
        for i in range(len(coords)):
            if used[i]:
                continue
            cluster = [coords[i]]
            used[i] = True
            for j in range(i + 1, len(coords)):
                if used[j]:
                    continue
                dist = np.linalg.norm(coords[i].astype(float) - coords[j].astype(float))
                if dist < self._branch_ks * 2:
                    cluster.append(coords[j])
                    used[j] = True
            centroid = np.mean(cluster, axis=0).astype(int)
            points.append((int(centroid[1]), int(centroid[0])))  # (u, v)

        return points

    # ------------------------------------------------------------------ #
    #  Temporal node stabilization
    # ------------------------------------------------------------------ #
    def _update_tracked_nodes(self, detected_pts):
        """Associate detected branch points with tracked nodes (nearest-neighbor)."""
        # Age all existing nodes
        for tn in self._tracked_nodes:
            tn['age'] += 1

        # Associate detections with nearest tracked node
        used_detections = set()
        for tn in self._tracked_nodes:
            best_dist = self._assoc_max_dist
            best_idx = -1
            for i, (u, v) in enumerate(detected_pts):
                if i in used_detections:
                    continue
                dist = np.sqrt((tn['u'] - u) ** 2 + (tn['v'] - v) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                tn['u'] = detected_pts[best_idx][0]
                tn['v'] = detected_pts[best_idx][1]
                tn['age'] = 0
                used_detections.add(best_idx)

        # Add unmatched detections as new nodes
        for i, (u, v) in enumerate(detected_pts):
            if i not in used_detections:
                self._tracked_nodes.append({
                    'id': self._next_node_id,
                    'u': u,
                    'v': v,
                    'age': 0,
                })
                self._next_node_id += 1

        # Remove stale nodes
        self._tracked_nodes = [
            tn for tn in self._tracked_nodes if tn['age'] <= self._node_max_age
        ]

    # ------------------------------------------------------------------ #
    #  3D projection
    # ------------------------------------------------------------------ #
    def _project_3d(self, u, v, stamp):
        """Project pixel (u,v) to 3D using depth and intrinsics."""
        if self._depth is None or self._intrinsics is None:
            return None

        fx, fy, cx, cy = self._intrinsics

        # Bounds check
        h, w = self._depth.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        z_mm = int(self._depth[v, u])
        if z_mm < self._depth_min or z_mm > self._depth_max:
            return None

        z = z_mm / 1000.0  # to meters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pt = PointStamped()
        pt.header.stamp = stamp
        pt.header.frame_id = 'oak_camera_frame'
        pt.point.x = float(x)
        pt.point.y = float(y)
        pt.point.z = float(z)
        return pt

    # ------------------------------------------------------------------ #
    #  Rule-based cut selection
    # ------------------------------------------------------------------ #
    def _select_cuts(self, nodes_msg, label_mask, stamp):
        """Select candidate cut targets based on simple rules."""
        cut_msg = CutPlan()
        cut_msg.header.stamp = stamp
        cut_msg.header.frame_id = 'oak_rgb_optical_frame'

        leaf_mask = (label_mask == LEAF).astype(np.uint8)
        h, w = label_mask.shape[:2]
        target_id = 0

        for node in nodes_msg.nodes:
            reasons = []
            conf_factors = []

            # Rule 1: below Y threshold (lower in image = lower on plant)
            if node.v > self._cut_y_thresh:
                reasons.append('NEAR_CROWN')
                conf_factors.append(0.6)

            # Rule 2: high local leaf density
            u, v = node.u, node.v
            r = self._cut_density_r
            y1 = max(0, v - r)
            y2 = min(h, v + r)
            x1 = max(0, u - r)
            x2 = min(w, u + r)
            region = leaf_mask[y1:y2, x1:x2]
            if region.size > 0:
                density = region.sum() / region.size
                if density > self._cut_density_thresh:
                    reasons.append('OVERCROWDING')
                    conf_factors.append(float(density))

            if reasons:
                ct = CutTarget()
                ct.id = target_id
                ct.point_camera = node.point_camera
                ct.confidence = float(np.mean(conf_factors)) if conf_factors else 0.0
                ct.reason_code = ','.join(reasons)

                if ct.confidence >= self._cut_min_conf:
                    cut_msg.targets.append(ct)
                    target_id += 1

        return cut_msg

    # ------------------------------------------------------------------ #
    #  RViz markers
    # ------------------------------------------------------------------ #
    def _publish_markers(self, nodes_msg, cut_msg, stamp):
        markers = MarkerArray()

        # Node markers (green spheres)
        for i, node in enumerate(nodes_msg.nodes):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'oak_camera_frame'
            m.ns = 'stem_nodes'
            m.id = node.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = node.point_camera.point.x
            m.pose.position.y = node.point_camera.point.y
            m.pose.position.z = node.point_camera.point.z
            m.pose.orientation.w = 1.0
            m.scale.x = 0.01
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = float(node.confidence)
            m.lifetime.sec = 1
            markers.markers.append(m)

        # Cut target markers (red spheres)
        for ct in cut_msg.targets:
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'oak_camera_frame'
            m.ns = 'cut_targets'
            m.id = ct.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = ct.point_camera.point.x
            m.pose.position.y = ct.point_camera.point.y
            m.pose.position.z = ct.point_camera.point.z
            m.pose.orientation.w = 1.0
            m.scale.x = 0.015
            m.scale.y = 0.015
            m.scale.z = 0.015
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.9
            m.lifetime.sec = 1
            markers.markers.append(m)

        self._pub_markers.publish(markers)

    # ------------------------------------------------------------------ #
    #  Debug overlay
    # ------------------------------------------------------------------ #
    def _make_overlay(self, label_mask, skeleton, branch_pts, cut_msg):
        overlay = self._rgb.copy()
        h, w = overlay.shape[:2]

        # Semi-transparent class overlay
        color_mask = np.zeros_like(overlay)
        for cls_id, color in CLASS_COLORS.items():
            color_mask[label_mask == cls_id] = color
        cv2.addWeighted(color_mask, 0.35, overlay, 0.65, 0, overlay)

        # Draw skeleton
        if skeleton is not None:
            skel_resized = skeleton
            if skeleton.shape[:2] != (h, w):
                skel_resized = cv2.resize(skeleton, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[skel_resized > 0] = (255, 255, 0)  # cyan skeleton

        # Draw branch points (green circles)
        for u, v in branch_pts:
            cv2.circle(overlay, (u, v), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (u, v), 7, (255, 255, 255), 1)

        # Draw cut targets (red X)
        for ct in cut_msg.targets:
            # Back-project to pixel if we have intrinsics
            cu, cv_pt = self._back_project(ct.point_camera)
            if cu is not None:
                sz = 8
                cv2.line(overlay, (cu - sz, cv_pt - sz), (cu + sz, cv_pt + sz), (0, 0, 255), 2)
                cv2.line(overlay, (cu - sz, cv_pt + sz), (cu + sz, cv_pt - sz), (0, 0, 255), 2)
                cv2.putText(
                    overlay, ct.reason_code, (cu + 10, cv_pt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                )

        return overlay

    def _back_project(self, pt_stamped):
        """Back-project 3D point to pixel coordinates for overlay."""
        if self._intrinsics is None:
            return None, None
        fx, fy, cx, cy = self._intrinsics
        z = pt_stamped.point.z
        if z <= 0:
            return None, None
        u = int(pt_stamped.point.x * fx / z + cx)
        v = int(pt_stamped.point.y * fy / z + cy)
        return u, v


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
