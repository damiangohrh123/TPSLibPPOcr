# legacy/

The HTTP-server era of this project. Nothing here is on the path to production.

`ocr_server` loads the OCR models once and serves them over `POST /ocr` on port 8080,
so a separate process can send it a raw BGR888 frame and get JSON back. That was how
OCR reached its callers before the pipeline was packaged as a library.

Production does not work this way. The board's state machine loads `libTPSLibPPOcr.so`
and calls it directly through a flat C ABI, so there is no server process, no HTTP, and
no localhost round trip. The library returns the same JSON shape `ocr_server` does,
which is deliberate: a caller parses one format whichever it talks to.

## Why it is still here

`recc_gen5_test_kit` reaches OCR through this server. `test_ocr_continuous.py` POSTs to
it, and `automation_driver.py` imports that module for its live captures. Removing
`ocr_server` today would leave the Python harness with no way to read a screen, and that
harness is where the rule matching, step sequencing and HID output are developed.

The eventual replacement is for the harness to load the library directly rather than
speak HTTP to a copy of the same pipeline. Until then this stays, and stays working.

## What is in here

- `ocr_server.cpp` the server itself: loads models once, serves `POST /ocr` and `/health`
- `http_server.cpp` / `.h` a minimal HTTP layer over POSIX sockets, used only by the above
- `alarm_detector.cpp` / `.h` HSV red-banner detection, and the `alarm` field in the
  server's response

Alarm detection is legacy for a second reason. The current design treats an alarm as one
kind of rule step, matched against the text OCR returns in the same way a click step is,
so the library has no alarm detector at all. Nothing reads the server's `alarm` field any
more. It costs about 12 ms per frame at 1024x768 and 21 ms at 1920x1080.

## Building

Built as part of the normal CMake build with `-DBUILD_TOOLS=ON`, against `ocr_core` from
`pipeline/ocr/`. It shares `api/json_write.cpp` with `step_matcher`, which is current code
and stays where it is.
