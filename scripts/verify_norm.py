import blobconverter, hashlib

blob_no_norm = blobconverter.from_onnx(
    model='exports/plant_seg_lraspp_640_opset11.onnx',
    data_type='FP16',
    shaves=6,
    output_dir='exports/test_no_norm/',
)
with open(blob_no_norm, 'rb') as f:
    h1 = hashlib.md5(f.read()).hexdigest()

blob_with_norm = blobconverter.from_onnx(
    model='exports/plant_seg_lraspp_640_opset11.onnx',
    data_type='FP16',
    shaves=6,
    output_dir='exports/test_with_norm/',
    optimizer_params=[
        '--reverse_input_channels',
        '--mean_values=[123.675,116.28,103.53]',
        '--scale_values=[58.395,57.12,57.375]',
        '--input_shape=[1,3,640,640]',
    ],
)
with open(blob_with_norm, 'rb') as f:
    h2 = hashlib.md5(f.read()).hexdigest()

print(f'Without norm: {h1}')
print(f'With norm:    {h2}')
same = (h1 == h2)
print(f'Same: {same}')
if not same:
    print('Normalization IS baked into the blob.')
else:
    print('WARNING: Blobs are identical - normalization may NOT be baked in.')
