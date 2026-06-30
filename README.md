# PPOCR — RK3588 Deployment

PP-OCRv6 tiny det + rec running on the Rockchip RK3588 NPU via RKNN.

| Component | Model | Format | Notes |
|-----------|-------|--------|-------|
| Det | PP-OCRv6 tiny det | INT8, 480×480 | ImageNet norm baked in; Python sends raw BGR |
| Rec | PP-OCRv6 tiny rec | FP16, 320×48 | [0,1] /255 normalisation |

## Directory Structure

```
ppocr/
  model/
    PP-OCRv6_tiny_det.onnx          # conversion source
    PP-OCRv6_tiny_rec.onnx          # conversion source
    PP-OCRv6_tiny_det_rk3588.rknn   # runtime model
    PP-OCRv6_tiny_rec_rk3588.rknn   # runtime model
    ppocr_keys_v6.txt               # character dictionary
  python/
    ppocr_det.py                    # det inference wrapper
    ppocr_rec.py                    # rec inference wrapper
    ppocr_system.py                 # det + rec pipeline
    ppocr_benchmark.py              # timing / accuracy benchmark
    tesseract_benchmark.py          # Tesseract comparison
    convert_v6_tiny_det.py          # ONNX → RKNN (det)
    convert_v6_tiny_rec.py          # ONNX → RKNN (rec)
    utils/                          # postprocessing utilities
```

## Requirements

**Board (RK3588):**
- `librknnrt 2.4.2a2+` (tested: 2.4.2a2)
- Python 3.8+, `rknn-toolkit-lite2`
- `opencv-python`, `numpy`, `shapely`, `pyclipper`

**Host (conversion only):**
- `rknn-toolkit2 2.4.2a8+` (tested: 2.4.2a8)

## Running the Benchmark

```bash
cd python
PPOCR_CHAR_DICT=../model/ppocr_keys_v6.txt \
python3 ppocr_benchmark.py \
  --image_path <path/to/image> \
  --cycles 10
```

Key arguments:

| Arg | Default | Description |
|-----|---------|-------------|
| `--image_path` | required | Input image |
| `--det_model` | `../model/PP-OCRv6_tiny_det_rk3588.rknn` | Det model path |
| `--rec_model` | `../model/PP-OCRv6_tiny_rec_rk3588.rknn` | Rec model path |
| `--cycles` | `1` | Number of inference cycles to average |
| `--input_scale` | `2.0` | Upscale factor before det |
| `--drop_score` | `0.4` | Min rec confidence to display |
| `--out_dir` | `annotated` | Output directory for annotated images |
| `--save_crops` | off | Save rec input crops to `--crops_dir` |
| `--crops_dir` | `crops` | Directory for saved crops (requires `--save_crops`) |

Key env vars: `PPOCR_CHAR_DICT`, `PPOCR_DET_THRESH`, `PPOCR_BOX_THRESH`, `PPOCR_UNCLIP_RATIO`, `PPOCR_DROP_SCORE`, `PPOCR_MIN_HEIGHT`, `PPOCR_MIN_WIDTH`

Note: `PPOCR_INPUT_SCALE` is read by `ppocr_system.py` when used as a standalone script, but `ppocr_benchmark.py` uses `--input_scale` instead.

## Re-converting Models

**Det** (INT8, calibration images in `dataset/`):
```bash
cd python
python3 convert_v6_tiny_det.py ../model/PP-OCRv6_tiny_det.onnx rk3588
```

**Rec** (FP16, no quantization):
```bash
cd python
python3 convert_v6_tiny_rec.py ../model/PP-OCRv6_tiny_rec.onnx rk3588
```

## Environment

| Item | Version |
|------|---------|
| RKNN Toolkit2 (host) | 2.4.2a8 |
| librknnrt (board) | 2.4.2a2 |
| Board | RK3588, Ubuntu 20.04 aarch64 |
| Board Python | 3.8 |
