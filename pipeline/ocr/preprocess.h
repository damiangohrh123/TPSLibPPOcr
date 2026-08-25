#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

// C++ port of the Python reference implementation's utils/preprocess.py.

// Applies (img * scale - mean) / std per channel. Shared by ppocr_det.cpp
// and ppocr_rec.cpp, each with its own scale/mean/std values.
// Only the HWC path is ported; Python's CHW branch was unused here too.
class NormalizeImage {
public:
    NormalizeImage(double scale, const std::array<double, 3>& mean, const std::array<double, 3>& std);

    // Returns a new image; img is left untouched.
    cv::Mat operator()(const cv::Mat& img) const;

private:
    double scale_;
    cv::Scalar mean_;
    cv::Scalar std_;
};

// Flattens an HWC CV_32FC3 image into a flat float32 buffer (batch size 1),
// the layout RknnExecutor::run() expects.
std::vector<float> to_nhwc_batch(const cv::Mat& img);
