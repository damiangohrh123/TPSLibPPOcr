#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ppocr_system.h"

// Text read from a frame or a region: the pieces in reading order and the text they join into.
struct Reading {
    std::string text;               // pieces on one line joined by spaces, lines by "\n", top to bottom
    double score = 0.0;             // mean piece score, 0 to 1
    std::vector<OcrResult> pieces;  // the pieces joined into text
};

// Joins results (already in reading order) whose box centre lies in keep; a centre more than half a box height from the previous one starts a new line.
Reading join_lines(const std::vector<OcrResult>& results, const cv::Rect& keep);

// Extends a crop to at least the detector's input size with its mean edge colour, so text keeps its native scale.
cv::Mat pad_to_det_input(const cv::Mat& crop);

// Reads a region drawn on a frame (Appendix E's padded detection): widens it by margin, pads it and keeps text centred inside it; boxes are in frame pixels.
Reading read_region(const TextSystem& system, const cv::Mat& frame, const cv::Rect& region, int margin);
