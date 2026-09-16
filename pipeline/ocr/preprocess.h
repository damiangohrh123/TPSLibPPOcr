#pragma once

#include <vector>

#include <opencv2/core.hpp>

// C++ port of the Python reference implementation's utils/preprocess.py.

// Multiplies an HWC 8-bit image by scale and returns it as float32. Shared by
// ppocr_det.cpp and ppocr_rec.cpp, each with its own scale.
class NormalizeImage {
public:
    explicit NormalizeImage(double scale);

    // Returns a new image; img is left untouched.
    cv::Mat operator()(const cv::Mat& img) const;

private:
    double scale_;
};

// Flattens an HWC CV_32FC3 image into a flat float32 buffer (batch size 1),
// the layout RknnExecutor::run() expects.
std::vector<float> to_nhwc_batch(const cv::Mat& img);
