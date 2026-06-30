"""
Convert PP-OCRv6 tiny rec ONNX to RKNN (FP16, no quantization).

Usage:
    python3 convert_v6_tiny_rec.py <onnx_path> <platform> [output_path]

Example:
    python3 convert_v6_tiny_rec.py ../model/PP-OCRv6_tiny_rec.onnx rk3588
"""
import sys
import os
from rknn.api import RKNN

if len(sys.argv) < 3:
    print('Usage: python3 convert_v6_tiny_rec.py <onnx_path> <platform> [output_path]')
    print('       platform: rk3588, rk3576, rk356x, rv1106, ...')
    sys.exit(1)

model_path  = sys.argv[1]
platform    = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else \
    '../model/PP-OCRv6_tiny_rec_{0}.rknn'.format(platform)

print('ONNX    :', model_path)
print('Platform:', platform)
print('Output  :', output_path)

rknn = RKNN(verbose=False)

print('--> Config model')
rknn.config(target_platform=platform)
print('done')

print('--> Loading model')
ret = rknn.load_onnx(model=model_path)
if ret != 0:
    print('Load model failed!')
    sys.exit(ret)
print('done')

print('--> Building model (FP16, no quantization)')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    sys.exit(ret)
print('done')

print('--> Export:', output_path)
ret = rknn.export_rknn(output_path)
if ret != 0:
    print('Export rknn model failed!')
    sys.exit(ret)
print('done')

rknn.release()
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f'\nSaved: {output_path}  ({size_mb:.1f} MB)')
