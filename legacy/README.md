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

## Deploying and running it

### Run it

```bash
cd ~/board_deploy
./ocr_server /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt 8080
```

Confirm it's up:

```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/health').read())"
```

`/ocr` requires the 8-byte width and height header before the pixel bytes, so it is not a plain `curl --data-binary @file` call. See `recc_gen5_test_kit/test_ocr_continuous.py` for a working example, or the wire format under Usage below.

### Run as a service (survives reboot)

`legacy/ocr_server.service` is a systemd unit that starts `ocr_server` on boot and restarts it if it crashes. `kvmd` has no equivalent yet. It is backgrounded once by `kvmd_run.sh`, with no restart-on-crash supervision; giving it the same systemd treatment is a future improvement.

```bash
sudo cp ~/board_deploy/ocr_server.service /etc/systemd/system/ocr_server.service  # copy it to the board first
sudo systemctl daemon-reload
sudo systemctl enable ocr_server
sudo systemctl start ocr_server
```

Check status or logs:

```bash
sudo systemctl status ocr_server --no-pager
journalctl -u ocr_server -f
```

### HTTP API

`ocr_server` loads the detection and recognition models once at startup, then serves OCR and alarm-detection results over HTTP, rather than once per invocation as `benchmark` does.

#### `GET /health`

Returns `200` with `{"status":"ok"}` once models are loaded.

#### `POST /ocr`

Request body: an 8-byte header followed by tightly-packed pixel data, no padding:

```
[4 bytes: width,  big-endian uint32]
[4 bytes: height, big-endian uint32]
[width * height * 3 bytes: BGR888 pixel data, row-major, 3 bytes/pixel]
```

This is the same byte order `kvmd`'s raw channel uses for its width, height, and size fields, so a client that already speaks that protocol, such as `recc_gen5_test_kit/test_ocr_continuous.py`, needs no second convention for this endpoint.

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

`alarm` is always present. The `bbox` and `text` fields inside it appear only when `detected` is `true`. `boxes` holds the same per-box text, score, and quad data `benchmark` prints, serialized as JSON.

Response `400` if the body is too short, or if its size does not match `8 + width*height*3` for the given width and height. The body is `{"error": "..."}` describing the mismatch.

## Current limitations

- One request is handled at a time, on the calling thread. There is no concurrency.
- No keep-alive, chunked encoding, or HTTPS. Every request opens a new connection.
- No authentication. Anyone who can reach the port can call it.
- The HTTP layer is a small implementation over POSIX sockets, not a vendored library.
