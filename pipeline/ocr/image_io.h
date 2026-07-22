#pragma once
#include <cstddef>
#include <string>
#include <opencv2/core.hpp>

// Used by benchmark.cpp for decoding images via stb_image; ocr_server.cpp
// decodes from an in-memory buffer instead (see decode_image_from_memory
// below), since it receives frames over HTTP rather than from disk.

// Decodes an image file to BGR (stb_image gives RGB; OpenCV expects BGR).
// Dispatches to load_raw_bgr888() instead for a ".bgr888" path. Returns an
// empty Mat on failure instead of exiting the process.
cv::Mat load_image(const std::string& path);

// Loads a raw, uncompressed BGR888 image: 3 bytes per pixel, blue-green-red
// order, no header, no compression -- the same bytes a CV_8UC3 cv::Mat holds
// in memory, so there's nothing to decode. Used to compare OCR against a
// lossy-compressed image with the same content, since JPEG/H.264 artifacts
// can distort small text. Raw pixel data carries no size information of its
// own, so the file name must encode it: ..._<width>x<height>.bgr888, e.g.
// alarm_1024x768.bgr888. Returns an empty Mat on failure.
cv::Mat load_raw_bgr888(const std::string& path);

// Decodes an in-memory image buffer (e.g. JPEG/PNG bytes from an HTTP
// request body) to BGR. Returns an empty Mat on failure.
cv::Mat decode_image_from_memory(const unsigned char* data, size_t len);
