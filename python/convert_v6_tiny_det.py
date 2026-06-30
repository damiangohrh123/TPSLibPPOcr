"""
Convert PP-OCRv6 tiny det ONNX to RKNN (INT8, normalization baked in).

Bakes ImageNet mean/std normalization into the RKNN model so Python sends raw
BGR pixels and RKNN handles normalization internally. INT8 avoids the FP16
ConvExGelu (GELU) NPU execution bug in RKNN 2.3.2. Calibration uses raw JPEG
images (NOT pre-normalized .npy) so the toolkit applies the baked-in
normalization during calibration.

Usage:
    python3 convert_v6_tiny_det.py <onnx_path> <platform> [output_path]

Example:
    python3 convert_v6_tiny_det.py ../model/PP-OCRv6_tiny_det.onnx rk3588
"""

import sys, os
from rknn.api import RKNN

if len(sys.argv) < 3:
    print('Usage: python3 convert_v6_tiny_det.py <onnx_path> <platform> [output_path]')
    sys.exit(1)

onnx_path   = sys.argv[1]
platform    = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else \
    '../model/PP-OCRv6_tiny_det_{0}.rknn'.format(platform)

print('ONNX    :', onnx_path)
print('Platform:', platform)
print('Output  :', output_path)

rknn = RKNN(verbose=True)

print('\n--> Config (baking in ImageNet normalization)')
# ImageNet mean/std in 0-255 scale; PaddleOCR convention: RGB means applied to BGR channels
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],
    std_values=[[58.395, 57.12, 57.375]],
    target_platform=platform,
)
print('done')

print('--> Load ONNX (fixing input to 1x3x480x480)')
ret = rknn.load_onnx(
    model=onnx_path,
    inputs=['x'],
    input_size_list=[[1, 3, 480, 480]]
)
if ret != 0:
    print('Load failed!')
    sys.exit(ret)
print('done')

DATASET_PATH = '../dataset/dataset_20.txt'
print(f'--> Build (INT8, calibrating with raw JPEGs from {DATASET_PATH})')
ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
if ret != 0:
    print('Build failed!')
    sys.exit(ret)
print('done')

print('--> Export:', output_path)
ret = rknn.export_rknn(output_path)
if ret != 0:
    print('Export failed!')
    sys.exit(ret)
print('done')

rknn.release()
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f'\nSaved: {output_path}  ({size_mb:.1f} MB)')
