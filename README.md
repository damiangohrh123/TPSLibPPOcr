# TPSLibPPOcr: OCR for the RECC Board (RK3588)

This repo turns one image of a machine's HMI screen into text: PP-OCRv6 tiny detection and recognition on the Rockchip RK3588 NPU via RKNN. An image goes in, and each piece of text found comes out with its position and confidence. Input is raw BGR888 only, because uncompressed pixel data preserves small text better.

| Component | Model | Format | Notes |
|-----------|-------|--------|-------|
| Det | PP-OCRv6 tiny det | INT8, 480×480 | ImageNet norm baked in; raw BGR in |
| Rec | PP-OCRv6 tiny rec | FP16, 320×48 | `[0,1]` normalisation |

Detection runs a full-image pass at the detector's fixed 480x480 input, then a grid of overlapping tiles closer to native resolution, merging both and dropping duplicates by NMS so small text survives the full-frame resize.

### Why There Is No Preprocessing

Preprocessing was tested on both stages and dropped. Binarization and upscaling made detection worse, so detection runs on unprocessed input. CLAHE contrast enhancement and unsharp-mask sharpening, applied to each crop before resizing, hurt recognition accuracy on every test screen: the 82.5% baseline fell to 59.6% with CLAHE and 77.4% with sharpening. Recognition therefore also runs on unprocessed crops.

Those figures come from the `test/*.bgr888` screens, which are screenshots of machine displays supplied for testing rather than frames captured from a machine over HDMI. They are the right input for comparing options against each other, since every option sees identical pixels, but the absolute numbers are not production-path figures: a real HDMI capture goes through the machine's video output and the board's HDMI receiver first. Nothing has yet been measured on a real HDMI capture of a real machine.

## Production Use

In production, OCR is meant to ship as `libTPSLibPPOcr.so` with a flat C ABI. This repo does not build that library yet; `benchmark` is the current way to run the pipeline on an image.

## Directory Structure

```
tpslibppocr/
  ocr/                             # the OCR pipeline
    benchmark.cpp                  # CLI: one .bgr888 image in, OCR text and timing out
    ppocr_det.cpp/h                # detection
    ppocr_rec.cpp/h                # recognition
    ppocr_system.cpp/h             # det + rec pipeline, NMS
    rknn_executor.cpp/h            # low-level RKNN model runner
    preprocess.cpp/h               # normalisation shared by both models
  test/                            # test screen images and sweep_det_thresholds.sh (see Usage below)
  cmake/aarch64-toolchain.cmake    # cross-compile toolchain file (see Build below)
  third_party/                     # rknn_api.h only (librknnrt.so lives on the board)
  aarch64-ubuntu20.04-toolchain.tar.gz  # cached aarch64 cross-compile toolchain (gitignored), see the section below
  model/
    PP-OCRv6_tiny_det.onnx         # conversion source
    PP-OCRv6_tiny_rec.onnx         # conversion source
    PP-OCRv6_tiny_det_rk3588.rknn  # runtime model
    PP-OCRv6_tiny_rec_rk3588.rknn  # runtime model
    ppocr_keys_v6.txt              # character dictionary
```

## Getting Started

Follow these steps in order: build once, deploy to a board, then run it.

### Prerequisites

- An aarch64 gcc-9 toolchain matching the board's OS (Ubuntu 20.04, glibc 2.31), plus statically-built OpenCV 4.5.4 (core+imgproc only), Clipper/polyclipping, and zlib for aarch64. `aarch64-ubuntu20.04-toolchain.tar.gz` in the repo root provides all of it pre-built, saving the roughly 20-minute bootstrap.
- `librknnrt.so` on the board itself. It is already present at `/usr/lib` as part of the board's OS image, so no installation is needed.

### 1. Build

The build produces the `ocr_core` library and, with `-DBUILD_TOOLS=ON`, the `benchmark` executable: an aarch64 binary, statically linked against OpenCV, Clipper, and zlib, leaving only `librknnrt.so` dynamic. Build output is not committed, to keep binary blobs out of git history, so a fresh clone needs a build before first use. Rebuild only when the C++ source changes.

Extract the cached toolchain once:

```bash
tar xzf aarch64-ubuntu20.04-toolchain.tar.gz -C ~
bash ~/fix_toolchain_paths.sh
```

This creates `~/cross` and `~/cross20` (the aarch64 toolchain), `~/deps-arm64` (OpenCV/Clipper/zlib), and `~/rknn-arm64` (RKNN SDK libs), and rewrites the archive's baked-in absolute paths to your `$HOME`. `fix_toolchain_paths.sh` is safe to re-run.

The archive already contains the wrapper compilers (`~/cross20/wrap/aarch64-linux-gnu-g++` and `-gcc`), which pass the correct `-B` flags for the target binutils and the gcc-9 frontend. `cmake/aarch64-toolchain.cmake` in this repo points at them.

Export `LD_LIBRARY_PATH` so the wrappers can find their own host libraries, then configure and build:

```bash
export LD_LIBRARY_PATH=$HOME/cross20/usr/lib/x86_64-linux-gnu:$HOME/cross/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Use $HOME, not ~: the shell does not expand a tilde inside a word that starts
# with -D, so cmake would receive the literal "~/deps-arm64/..." and the build
# would fail later on a missing opencv2/core.hpp.
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
      -DOPENCV_INCLUDE_DIR=$HOME/deps-arm64/include/opencv4 -DOPENCV_LIB_DIR=$HOME/deps-arm64/lib \
      -DCLIPPER_INCLUDE_DIR=$HOME/deps-arm64/include -DCLIPPER_LIB_DIR=$HOME/deps-arm64/lib \
      -DZLIB_LIB_DIR=$HOME/deps-arm64/lib -DRKNN_SDK_LIB_DIR=$HOME/rknn-arm64/lib \
      -DBUILD_TOOLS=ON
cmake --build build -j
```

### 2. Deploy to a Board

New board:

1. Copy the test images, model files and binary over (from Windows/WSL). Copy folders into the parent folder: naming a folder that does not exist yet as the destination makes `pscp` fail with `unable to open`.

   ```bash
   pscp -r test tpsadmin@<board-ip>:/home/tpsadmin/
   pscp -r model tpsadmin@<board-ip>:/home/tpsadmin/
   pscp build/benchmark tpsadmin@<board-ip>:/home/tpsadmin/test/
   ```

2. Test manually before relying on it. Run `benchmark` against one of the test images. This catches a wrong path or a missing model file in plain view.

Existing board, after a rebuild: re-upload just the binary.

```bash
pscp build/benchmark tpsadmin@192.168.1.101:/home/tpsadmin/test/
```

## Usage

### `benchmark` (One-Shot CLI)

Loads a raw `.bgr888` frame, runs detection and recognition, prints the results, and reports timing, CPU, and memory. Its detection defaults match the production values (det_thresh 0.2, box_thresh 0.4, unclip_ratio 1.5, max_candidates 3000). Pass alternatives positionally to sweep.

```bash
cd ~/test
./benchmark /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt alarm_1024x768.bgr888
```

`test/` also has `alarm_1920x1080.bgr888`, `auto_mode_1_1024x768.bgr888`, `auto_mode_2_1024x768.bgr888`, `normal_run_1024x768.bgr888`, and `full_test_1024x384.bgr888`. Pass a cycle count to average the timing, CPU, and memory numbers over repeated runs, for example `... alarm_1024x768.bgr888 10`.

The detector's own thresholds (`det_thresh`, `box_thresh`, `unclip_ratio`, `max_candidates`) are also CLI-configurable, as optional positional args after cycles and drop_score: `... alarm_1024x768.bgr888 1 0.4 <det_thresh> <box_thresh> <unclip_ratio> <max_candidates>`. These had been hardcoded since the project's first commit, with no record of being tested against alternatives. `test/sweep_det_thresholds.sh` sweeps a small grid of them against the 1024-wide test images and logs box counts, timing, and recognized text per combination to `sweep_results.csv`. Run it from `~/test` on the board, with `benchmark` copied in.

## Environment

| Item | Version |
|------|---------|
| RKNN Toolkit2 (host, for model conversion) | 2.4.2a8 |
| librknnrt (board) | 2.4.2a2+ (tested: 2.4.2a2) |
| OpenCV / Clipper / zlib (board) | statically linked, core+imgproc only (see `CMakeLists.txt`) |
| Board | RK3588, Ubuntu 20.04 aarch64 |
| Board IP (this dev board) | 192.168.1.101, user tpsadmin |
| Board deploy paths | `/home/tpsadmin/test` (benchmark and test images), `/home/tpsadmin/model` |
