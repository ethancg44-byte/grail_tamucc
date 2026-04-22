#!/usr/bin/env bash
# Export ONNX model to OpenVINO IR (FP16) with normalization baked in.
#
# This script uses OpenVINO Model Optimizer (mo) to convert the ONNX model
# into OpenVINO IR format (.xml + .bin) with:
#   - FP16 precision (required for OAK-D Pro / Myriad X)
#   - BGR→RGB channel reversal baked into the model
#   - ImageNet normalization baked in (OAK sends raw uint8, no preprocessing needed)
#
# Prerequisites:
#   pip install openvino-dev
#
# Usage:
#   ./export_to_openvino.sh <input_onnx> [output_dir]
#
# Example:
#   ./export_to_openvino.sh exports/plant_seg_lraspp.onnx exports/openvino

set -euo pipefail

# --- Arguments ---
ONNX_PATH="${1:?Usage: $0 <input.onnx> [output_dir]}"
OUTPUT_DIR="${2:-exports/openvino}"

# --- Validate input ---
if [ ! -f "$ONNX_PATH" ]; then
    echo "ERROR: ONNX file not found: $ONNX_PATH"
    exit 1
fi

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

MODEL_NAME="plant_seg_lraspp"

echo "=== OpenVINO Model Optimizer ==="
echo "  Input:    $ONNX_PATH"
echo "  Output:   $OUTPUT_DIR/${MODEL_NAME}.xml"
echo "  Precision: FP16"
echo ""

# --- Run Model Optimizer ---
# --reverse_input_channels: OAK sends BGR, model expects RGB → bake swap into IR
# --mean_values: ImageNet BGR means (note: reversed because of --reverse_input_channels)
#   Original RGB means: [123.675, 116.28, 103.53] → BGR order: [103.53, 116.28, 123.675]
#   But since --reverse_input_channels reverses BEFORE mean subtraction,
#   we provide the values in RGB order and mo handles the rest.
# --scale_values: ImageNet RGB stds × 255 = [58.395, 57.12, 57.375]
#
# Input: uint8 BGR [0-255] from OAK
# After --reverse_input_channels: uint8 RGB [0-255]
# After --mean_values: centered float
# After --scale_values: normalized (equivalent to (x/255 - mean) / std)

mo \
    --input_model "$ONNX_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --compress_to_fp16 \
    --reverse_input_channels \
    --mean_values "[123.675,116.28,103.53]" \
    --scale_values "[58.395,57.12,57.375]" \
    --input_shape "[1,3,768,768]"

echo ""
echo "=== Export complete ==="
echo "  XML: $OUTPUT_DIR/${MODEL_NAME}.xml"
echo "  BIN: $OUTPUT_DIR/${MODEL_NAME}.bin"
echo ""
echo "Next step: compile to .blob with compile_blob.sh"
