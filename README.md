# GRAIL — Greenhouse Robotics for Automated Intelligent Labor

## Plant Perception System — ROS 2 Jazzy on Raspberry Pi 5

Vertical greenhouse plant-structure perception using a Luxonis OAK-D Pro camera with on-device semantic segmentation. Designed for future integration with a PincherX-100 (px100) robotic arm for automated pruning.

## Architecture

```
OAK-D Pro ──> oak_depthai_wrapper ──> plant_perception ──> Cut Plan
  RGB + Depth     DepthAI pipeline     Skeletonize          ↓
  NN .blob         + mock mode         Branch detect    px100_integration
                                       3D project       (reachability)
```

### Packages

| Package | Description |
|---------|-------------|
| `plant_interfaces` | Custom message definitions (OrganSegmentation, StemNode, StemNodes, CutTarget, CutPlan) |
| `oak_depthai_wrapper` | DepthAI pipeline node — publishes RGB, depth, CameraInfo, NN output |
| `plant_perception` | Perception pipeline — segmentation decoding, skeletonization, node detection, 3D projection, cut planning |
| `px100_integration` | Optional dry-run reachability checker for PincherX-100 |
| `bringup` | Launch files, YAML configs, RViz config, TF setup |

## Setup Instructions

### 1. Install ROS 2 Jazzy (Ubuntu 24.04 arm64)

```bash
sudo apt update && sudo apt install -y software-properties-common curl

# Add ROS 2 repo
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-jazzy-ros-base ros-jazzy-rviz2 \
  ros-jazzy-cv-bridge ros-jazzy-image-transport \
  ros-jazzy-tf2-ros ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-visualization-msgs \
  ros-jazzy-rosidl-default-generators ros-jazzy-rosidl-default-runtime \
  python3-colcon-common-extensions python3-rosdep python3-vcstool \
  build-essential

echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

### 2. Install colcon and rosdep

```bash
sudo rosdep init  # skip if already done
rosdep update
```

### 3. Install Python dependencies

```bash
pip3 install numpy opencv-python-headless
```

### 4. Install DepthAI (for real OAK hardware)

```bash
# Install DepthAI Python library
pip3 install depthai

# Install udev rules for USB access
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

> **Note:** DepthAI is only required for real hardware. The system runs in mock mode without it.

### 5. Build the workspace

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 6. Verify build

```bash
# List all packages
ros2 pkg list | grep -E "plant_|oak_|px100|bringup"
```

Expected output:
```
bringup
oak_depthai_wrapper
plant_interfaces
plant_perception
px100_integration
```

## Running

### Perception with mock NN (no hardware needed)

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch bringup bringup_perception.launch.py
```

In another terminal, verify topics:
```bash
source ~/ros2_ws/install/setup.bash
ros2 topic list
```

Expected topics:
```
/oak/rgb/image_raw
/oak/depth_aligned/image_raw
/oak/camera_info
/oak/nn/segmentation_raw
/perception/label_mask
/perception/segmentation
/perception/nodes
/perception/cut_plan
/perception/node_markers
/debug/overlay_image
```

### Perception with real OAK-D Pro

1. Connect OAK-D Pro via USB 3.0
2. Verify device is detected:
   ```bash
   python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
   ```
3. Edit `src/bringup/config/oak_params.yaml`:
   ```yaml
   oak_camera_node:
     ros__parameters:
       enable_mock_nn: false
       model_blob_path: '/path/to/your/model.blob'
   ```
4. Rebuild and launch:
   ```bash
   colcon build --packages-select bringup --symlink-install
   source install/setup.bash
   ros2 launch bringup bringup_perception.launch.py
   ```

### With reachability checker

```bash
ros2 launch bringup bringup_with_reachability.launch.py
```

### RViz visualization

```bash
rviz2 -d ~/ros2_ws/src/bringup/rviz/perception.rviz
```

## Topics Reference

| Topic | Type | Description |
|-------|------|-------------|
| `/oak/rgb/image_raw` | `sensor_msgs/Image` | RGB image (BGR8) |
| `/oak/depth_aligned/image_raw` | `sensor_msgs/Image` | Depth aligned to RGB (16UC1, millimeters) |
| `/oak/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/oak/nn/segmentation_raw` | `sensor_msgs/Image` | NN segmentation output (mono8 labels) |
| `/perception/label_mask` | `sensor_msgs/Image` | Label mask (mono8: 0=BG, 1=LEAF, 2=STEM) |
| `/perception/segmentation` | `plant_interfaces/OrganSegmentation` | Segmentation metadata |
| `/perception/nodes` | `plant_interfaces/StemNodes` | Detected stem nodes with 3D positions |
| `/perception/cut_plan` | `plant_interfaces/CutPlan` | Recommended cut targets |
| `/perception/node_markers` | `visualization_msgs/MarkerArray` | RViz markers for nodes and cuts |
| `/debug/overlay_image` | `sensor_msgs/Image` | RGB with segmentation overlay and markers |

## TF Frames

```
oak_camera_frame
  ├── oak_rgb_optical_frame
  ├── oak_depth_optical_frame
  └── px100/base_link (placeholder)
```

## Segmentation Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | BACKGROUND | Background / non-plant |
| 1 | LEAF | Leaf tissue |
| 2 | STEM_OR_PETIOLE | Stem and petiole structures |

Future classes: BULB, CROWN (hooks in place).

## Training & Exporting the Segmentation Model

The perception pipeline uses an LRASPP-MobileNetV3-Large model (3 classes: BG, LEAF, STEM) that runs on the OAK-D Pro's Myriad X VPU.

### Quick Start (on GPU workstation)

```bash
# Install training dependencies
pip install -r requirements-training.txt

# Prepare dataset
python scripts/datasets/synthetic_plants.py \
  --raw_dir data/synthetic_plants_raw \
  --output_dir data/synthetic_plants

# Train + export ONNX
python scripts/train_segmentation.py \
  --data_dirs data/synthetic_plants \
  --output_dir checkpoints/run01 \
  --epochs 50 --export_onnx

# Convert to OpenVINO IR
./scripts/export_to_openvino.sh exports/plant_seg_lraspp.onnx exports/openvino

# Compile to DepthAI blob
./scripts/compile_blob.sh exports/openvino/plant_seg_lraspp.xml exports 6

# Deploy blob to Pi 5 and update oak_params.yaml
```

For the full step-by-step guide, see [docs/model_export.md](docs/model_export.md).

### Model Config

Model metadata and export parameters are documented in `models/lraspp_mobilenetv3/config.yaml`.

## PincherX-100 (Optional)

The `px100_integration` package provides a dry-run reachability checker only. It does **not** require Interbotix packages to build or run.

For full arm integration (motion planning, MoveIt), see Phase 3 of the project plan.

## License

MIT
