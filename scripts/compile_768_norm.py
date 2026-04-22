import blobconverter, hashlib, shutil

blob_with_norm = blobconverter.from_onnx(
    model='exports/plant_seg_lraspp_opset11.onnx',
    data_type='FP16',
    shaves=6,
    output_dir='exports/test_768_norm/',
    optimizer_params=[
        '--reverse_input_channels',
        '--mean_values=[123.675,116.28,103.53]',
        '--scale_values=[58.395,57.12,57.375]',
        '--input_shape=[1,3,768,768]',
    ],
)

with open(blob_with_norm, 'rb') as f:
    h = hashlib.md5(f.read()).hexdigest()
print(f'768 with norm: {h}')

shutil.copy2(blob_with_norm, 'exports/plant_seg_lraspp.blob')
print('Copied to exports/plant_seg_lraspp.blob')
