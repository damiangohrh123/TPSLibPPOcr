# OCR Pipeline — RK3588 (RECC)

PP-OCRv6 tiny detection + recognition running on the Rockchip RK3588 NPU via RKNN, reading text off machine HMI screens. Includes alarm-banner detection built on top of the raw OCR output. This repo covers the production C++ implementation only. The original Python reference implementation has been moved out into its own standalone project (no longer part of this repo).

| Component | Model | Format | Notes |
|-----------|-------|--------|-------|
| Det | PP-OCRv6 tiny det | INT8, 480×480 | ImageNet norm baked in; raw BGR in |
| Rec | PP-OCRv6 tiny rec | FP16, 320×48 | `[0,1]` normalisation |

Detection always runs a full-image "squash" pass, plus a grid of overlapping tiles sized to the detector's native 480x480 input, covering the whole image regardless of its aspect ratio — this avoids the resolution loss that comes from downscaling a much larger image straight to 480x480 in one shot. Alarm detection converts the frame to HSV, masks for red, and filters by area/aspect-ratio/screen-position to find alarm banners; if one is found, the OCR text already collected for that region is returned immediately without a second inference pass. Detection has no image preprocessing step (no binarization, no upscaling) — this was benchmarked against the actually-deployed detection model and found to perform best as-is.

## Directory Structure

```
recc/
  pipeline/                        # the two-stage OCR + decision/control system
    ocr/                           # Stage 1: OCR + alarm detection pipeline
      benchmark.cpp                # program entry point; also reports timing/CPU/memory with cycles > 1
      ppocr_det.cpp/h              # detection
      ppocr_rec.cpp/h              # recognition
      ppocr_system.cpp/h           # det + rec pipeline, tiling, NMS
      alarm_detector.cpp/h         # HSV-based alarm banner detection
      text_correction.cpp/h        # narrow post-processing fix for two known recognition mistakes
      rknn_executor.cpp/h          # low-level RKNN model runner
      preprocess.cpp/h
      image_io.cpp/h               # image decode helpers, used by benchmark and ocr_server
    automation/                    # Stage 2 prototype: matches a hand-written rule against one
                                    # saved OCR result and prints the resulting KVM action sequence
                                    # (simpler than the goal/step closed loop design, see README below)
  api/                             # Internal OCR API (ocr_server + minimal HTTP layer, see README below)
  third_party/rknn/                # RKNN SDK headers + librknnrt.so
  third_party/stb/                 # stb_image, used by benchmark and ocr_server
  cross_toolchain/                 # cached aarch64 cross-compile toolchain, see README below
  board_deploy/                    # built binaries + sample images, ready to pscp to the board
  model/
    PP-OCRv6_tiny_det.onnx         # conversion source
    PP-OCRv6_tiny_rec.onnx         # conversion source
    PP-OCRv6_tiny_det_rk3588.rknn  # runtime model
    PP-OCRv6_tiny_rec_rk3588.rknn  # runtime model
    ppocr_keys_v6.txt              # character dictionary
```

## Building and Running the C++ Port

`board_deploy/benchmark` is a pre-built, ready-to-run aarch64 binary — cross-compiled for the RK3588 board and statically linked against OpenCV 4.5.4 (core+imgproc only), Clipper/polyclipping, and zlib, so the board doesn't need any of those installed. The only runtime dependency it pulls in besides standard libc/libm/libpthread is `librknnrt.so`, which the board already provides at `/usr/lib`.

There is no cross-compile toolchain or aarch64-built OpenCV/Clipper/zlib set up on the board or dev machine — the pre-built binary above was produced by asking an AI assistant (Claude) to cross-compile it in its own sandboxed environment, since setting up a full aarch64 cross toolchain locally is a lot of one-time overhead. That environment (aarch64 gcc-9 toolchain matching the board's Ubuntu 20.04/glibc 2.31, plus statically-built OpenCV/Clipper/zlib) is cached in `cross_toolchain/aarch64-ubuntu20.04-toolchain.tar.gz` so a future rebuild skips the ~20+ minute bootstrap (steps to reuse it below). **If the C++ source changes, ask Claude to rebuild `board_deploy/benchmark` from the updated source** rather than trying to build locally, unless a proper local cross-compile environment gets set up (in which case, use the `cmake` invocation below and skip this whole workaround).

### Reusing the Cached Cross-Toolchain

`cross_toolchain/aarch64-ubuntu20.04-toolchain.tar.gz` contains an aarch64 cross-compiler matching the board's own toolchain era (gcc-9 / glibc 2.31 from Ubuntu 20.04 focal, plus host-side binutils) and statically-built dependencies for aarch64 (zlib, Clipper/polyclipping, and OpenCV 4.5.4 core+imgproc only).

1. Extract: `tar xzf aarch64-ubuntu20.04-toolchain.tar.gz -C ~` (creates `~/cross`, `~/cross20`, `~/deps-arm64`, `~/rknn-arm64`, and `~/fix_toolchain_paths.sh`).
2. Run `bash ~/fix_toolchain_paths.sh` once, right after extracting. A few files inside the archive (a GNU ld linker script inside `cross20/usr/aarch64-linux-gnu/lib/libc.so`, plus CMake/pkgconfig prefix variables for the arm64 OpenCV/zlib deps) have absolute paths baked in from whatever machine/environment originally built the toolchain. This script rewrites them to the current `$HOME` so the archive works regardless of where it's extracted. Safe to re-run; it's a no-op if paths already match.
3. Build the wrapper compilers pointing `-B` at `~/cross/usr/aarch64-linux-gnu/bin` (target binutils) and `~/cross20/usr/bin` (gcc-9 frontend) — see `~/cross20/wrap/aarch64-linux-gnu-g++` inside the archive for the exact form.
4. Set `LD_LIBRARY_PATH` to include `~/cross20/usr/lib/x86_64-linux-gnu` and `~/cross/usr/lib/x86_64-linux-gnu` (host-side private shared libs the compiler/assembler need).
5. Configure the project with `-DCMAKE_TOOLCHAIN_FILE=...` pointing at a toolchain file using those wrapper compilers, and `-DOPENCV_INCLUDE_DIR=~/deps-arm64/include/opencv4`, `-DOPENCV_LIB_DIR=~/deps-arm64/lib`, `-DCLIPPER_INCLUDE_DIR=~/deps-arm64/include`, `-DCLIPPER_LIB_DIR=~/deps-arm64/lib`, `-DZLIB_LIB_DIR=~/deps-arm64/lib`, `-DRKNN_SDK_LIB_DIR=~/rknn-arm64/lib`, `-DBUILD_TOOLS=ON`.

Building from source (once you have a real cross-compile toolchain + aarch64-built OpenCV/Clipper/zlib):

```bash
mkdir build && cd build
cmake .. -DOPENCV_INCLUDE_DIR=... -DOPENCV_LIB_DIR=... -DCLIPPER_INCLUDE_DIR=... \
         -DCLIPPER_LIB_DIR=... -DZLIB_LIB_DIR=... -DRKNN_SDK_LIB_DIR=... \
         -DBUILD_TOOLS=ON
make
```

`BUILD_TOOLS=ON` additionally builds `benchmark` — decodes an image, runs detection + recognition + alarm detection, prints results, and (with `cycles` > 1) reports timing/CPU/memory measurement, same methodology as the Python reference implementation's benchmark script. Requires `RKNN_SDK_LIB_DIR` pointing at a real `librknnrt.so`, since it links the actual board runtime.

On the board:

```bash
cd ~/board_deploy
chmod +x benchmark
./benchmark /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt testdata/alarm.jpg
```

`testdata/` also has `auto_mode_1.jpg`, `auto_mode_2.jpg`, `normal_run.jpg`, and `full_test.jpeg`. Add a cycle count to get timing/CPU/memory stats, e.g. `... testdata/alarm.jpg 10`.

Each of those also has a `..._<width>x<height>.bgr888` counterpart, e.g. `alarm_1024x768.bgr888`, the same picture with zero compression: raw BGR bytes, no JPEG encode/decode step at all. `load_image()` dispatches on the `.bgr888` extension automatically, no other command-line change needed, so the same `benchmark` binary can compare OCR against a lossy-compressed image and its lossless raw equivalent side by side. This exists because JPEG/H.264 artifacts can distort small text, and the team is evaluating whether feeding OCR raw BGR888 frames (skipping compression entirely) improves recognition on small fonts, at the cost of a much larger per-frame size, e.g. 1024x768 raw is 2.36MB versus ~150KB for the same picture as JPEG.

After a rebuild, re-upload the binary from Windows/WSL:

```bash
pscp board_deploy/benchmark tpsadmin@192.168.1.101:/home/tpsadmin/board_deploy/benchmark
```

See the Internal OCR API section below for running `ocr_server` (the HTTP API) on the board instead.

## Internal OCR API

`ocr_server` (`api/`) loads the detection/recognition models once at startup, then serves OCR and alarm-detection results over HTTP instead of running once per invocation like `ocr`/`benchmark`.

### `GET /health`

Returns `200` with `{"status":"ok"}` once models are loaded.

### `POST /ocr`

Request body: raw image bytes (JPEG/PNG).

Response `200`:

```json
{
  "boxes": [
    { "text": "ALARM 0089", "score": 0.97, "box": [[102,18],[210,18],[210,40],[102,40]] }
  ],
  "alarm": {
    "detected": true,
    "bbox": [80, 10, 900, 60],
    "text": "ALARM 0089 Kerf check: off center Z-EN"
  },
  "timing_ms": 412.3
}
```

`alarm` is always present; `bbox`/`text` inside it are only present when `detected` is `true`. `boxes` is the same per-box text/score/quad data `benchmark` already prints, just serialized instead of `printf`'d.

Response `400` if the body couldn't be decoded as an image: `{"error":"could not decode image"}`.

### Running It

On the board:

```bash
cd ~/board_deploy
chmod +x ocr_server
./ocr_server /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt 8080
```

From another shell on the board, or from a host machine that can reach the board:

```bash
curl http://192.168.1.101:8080/health
curl -X POST --data-binary @testdata/alarm.jpg http://192.168.1.101:8080/ocr
```

After a rebuild, re-upload the binary from Windows/WSL:

```bash
pscp board_deploy/ocr_server tpsadmin@192.168.1.101:/home/tpsadmin/board_deploy/ocr_server
```

### Current Limitations

- One request handled at a time, on the calling thread — no concurrency.
- No keep-alive, chunked encoding, or HTTPS — every request opens a new connection.
- No authentication — anyone who can reach the port can call it.
- The HTTP layer (`http_server.h/.cpp`) is a small implementation over POSIX sockets, not a vendored library.

## Rule-Based Automation (Prototype)

`pipeline/automation/` matches a person-written rule (a keyword to look for and what to do once found) against one saved OCR result, via fuzzy keyword matching, and prints the resulting KVM action sequence for a person to approve. It never touches an image, only the text and coordinates OCR already produced, so unlike everything else in `pipeline/`, it has no OpenCV/RKNN dependency and isn't gated behind `BUILD_TOOLS` or a cross-compile toolchain — it builds and runs on any machine with a C++17 compiler:

```bash
g++ -std=c++17 -O2 -I. pipeline/automation/*.cpp -o automation_runner
./automation_runner
```

It reads every rule from `rules/` and every real OCR result from `ocr_output/` (both plain JSON files, see `pipeline/automation/rule.h` and `pipeline/automation/screens.h` for the shapes), and matches each rule against each screen. Nothing is sent to a machine — the resulting sequence is printed and waits for a person to approve it.

This is a prototype of the fuller goal/step design described in `AI_automation_pipeline.docx` (Section 2), not a complete implementation of it: a rule here is a flat list of `{keyword, action, value}` steps, checked once against one already-saved screen, rather than the closed loop that design calls for — re-reading the screen via OCR after every action, retrying on a mismatch, and running live against a real KVM connection. Section 2.4 of that document lists exactly what's left to build to close that gap.

## Known Accuracy Limitations

An accuracy pass across real HMI screenshots found two recurring recognition-level error patterns: decimal points occasionally read as dashes (e.g. `0.000` → `0-000`), and "l"/"I"/"1" character confusion (e.g. `lines` → `Iines`). Everything else — fixed labels, status words, device IDs — reads reliably.

`text_correction.h/.cpp` now applies a narrow, hand-rolled fix for both patterns to every recognized box, inside `TextSystem::run()` so `benchmark` and `ocr_server` both get it automatically. Decimal-dash correction only touches a token that is entirely digits with exactly one dash and clean boundaries on both sides, so multi-dash device IDs (`ADAU1787-AD3-SCDD-CB-HBEA`) are left alone. The l/I/1 fix only snaps a word to a match against a small, fixed vocabulary of this project's actual recurring UI labels (`Full`, `Channel`, `Ch1`-`Ch4`, etc.), not general spell-checking — extend `known_vocabulary()` as new recurring labels are confirmed from real screens. This is a mitigation for known, well-evidenced mistakes, not a fix for the underlying recognition model; it remains the known weak point for a future recognition-model fine-tune.

## Requirements

**Board (RK3588):**
- `librknnrt 2.4.2a2+` (tested: 2.4.2a2)
- Statically-linked OpenCV core+imgproc, Clipper, zlib (see `CMakeLists.txt`)

## Environment

| Item | Version |
|------|---------|
| RKNN Toolkit2 (host, for model conversion) | 2.4.2a8 |
| librknnrt (board) | 2.4.2a2 |
| Board | RK3588, Ubuntu 20.04 aarch64 |
| Board IP | 192.168.1.101, user tpsadmin |
| Board deploy path | `/home/tpsadmin/board_deploy` (C++) |
