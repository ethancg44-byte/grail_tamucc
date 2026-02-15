#!/usr/bin/env bash
# Compile OpenVINO IR to DepthAI .blob for OAK-D Pro (Myriad X).
#
# Uses blobconverter Python package which handles compilation via
# Luxonis cloud service (no local OpenVINO compile_tool needed).
#
# Prerequisites:
#   pip install blobconverter
#
# Usage:
#   ./compile_blob.sh <openvino_xml> [output_dir] [shaves]
#
# Example:
#   ./compile_blob.sh exports/openvino/plant_seg_lraspp.xml exports 6
#
# The .blob file can then be deployed to the Pi 5 and referenced in
# oak_params.yaml:
#   model_blob_path: '/path/to/plant_seg_lraspp.blob'
#   enable_mock_nn: false

set -euo pipefail

# --- Arguments ---
XML_PATH="${1:?Usage: $0 <model.xml> [output_dir] [shaves]}"
OUTPUT_DIR="${2:-exports}"
SHAVES="${3:-6}"

# --- Validate input ---
if [ ! -f "$XML_PATH" ]; then
    echo "ERROR: OpenVINO XML not found: $XML_PATH"
    exit 1
fi

BIN_PATH="${XML_PATH%.xml}.bin"
if [ ! -f "$BIN_PATH" ]; then
    echo "ERROR: OpenVINO BIN not found: $BIN_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

MODEL_NAME="plant_seg_lraspp"
BLOB_PATH="$OUTPUT_DIR/${MODEL_NAME}.blob"

echo "=== Compiling .blob for OAK-D Pro (Myriad X) ==="
echo "  Input XML:  $XML_PATH"
echo "  Input BIN:  $BIN_PATH"
echo "  Shaves:     $SHAVES"
echo "  Output:     $BLOB_PATH"
echo ""

# --- Method 1: blobconverter (cloud compilation, recommended) ---
python3 -c "
import blobconverter
import sys

blob_path = blobconverter.from_openvino(
    xml='${XML_PATH}',
    bin='${BIN_PATH}',
    data_type='FP16',
    shaves=${SHAVES},
    output_dir='${OUTPUT_DIR}',
)
print(f'Blob compiled: {blob_path}')
"

echo ""
echo "=== Compilation complete ==="
echo "  Blob: $BLOB_PATH"
echo ""
echo "=== Deployment ==="
echo "  1. Copy .blob to Pi 5:"
echo "     scp $BLOB_PATH pi@<pi-ip>:~/ros2_ws/exports/"
echo ""
echo "  2. Update oak_params.yaml:"
echo "     model_blob_path: '~/ros2_ws/exports/${MODEL_NAME}.blob'"
echo "     enable_mock_nn: false"
echo ""
echo "  3. Rebuild and launch:"
echo "     cd ~/ros2_ws && colcon build --packages-select bringup --symlink-install"
echo "     source install/setup.bash"
echo "     ros2 launch bringup bringup_perception.launch.py"

# --- Method 2: Local compilation (alternative, requires OpenVINO compile_tool) ---
# Uncomment below if you prefer local compilation:
#
# COMPILE_TOOL=$(find /opt/intel/openvino* -name compile_tool -type f 2>/dev/null | head -1)
# if [ -z "$COMPILE_TOOL" ]; then
#     echo "ERROR: compile_tool not found. Install OpenVINO toolkit."
#     exit 1
# fi
# "$COMPILE_TOOL" \
#     -m "$XML_PATH" \
#     -d MYRIAD \
#     -VPU_NUMBER_OF_SHAVES "$SHAVES" \
#     -VPU_NUMBER_OF_CMX_SLICES "$SHAVES" \
#     -o "$BLOB_PATH"
