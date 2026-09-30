#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <opencv2/core.hpp>

// Raw BGR888 only: JPEG artifacts distort small text and production only ever captures raw.

// Parses "..._<width>x<height>.bgr888" -> {width, height}, or {0, 0} if the name doesn't match.
inline std::pair<int, int> parse_raw_dimensions(const std::string& path) {
    size_t ext = path.rfind(".bgr888");
    if (ext == std::string::npos) return {0, 0};
    size_t underscore = path.rfind('_', ext);
    if (underscore == std::string::npos) return {0, 0};
    std::string dims = path.substr(underscore + 1, ext - underscore - 1);
    size_t x = dims.find('x');
    if (x == std::string::npos) return {0, 0};
    return {std::atoi(dims.substr(0, x).c_str()), std::atoi(dims.substr(x + 1).c_str())};
}

// Reads the file's bytes straight into a CV_8UC3 Mat; returns an empty Mat on failure, having printed why.
inline cv::Mat load_image(const std::string& path) {
    auto [width, height] = parse_raw_dimensions(path);
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "failed to load %s: only raw \"..._<width>x<height>.bgr888\" "
            "files are supported\n", path.c_str());
        return cv::Mat();
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        return cv::Mat();
    }
    cv::Mat img(height, width, CV_8UC3);
    size_t expected_bytes = img.total() * img.elemSize();
    f.read(reinterpret_cast<char*>(img.data), expected_bytes);
    if (!f) {
        fprintf(stderr, "failed to read %s: expected %zu bytes for %dx%d\n",
            path.c_str(), expected_bytes, width, height);
        return cv::Mat();
    }
    return img;
}
