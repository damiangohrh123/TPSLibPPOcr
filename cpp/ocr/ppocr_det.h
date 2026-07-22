#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "rknn_executor.h"
#include "preprocess.h"

// C++ port of the Python reference implementation's ppocr_det.py.

// A detected quadrilateral: 4 corners ordered top-left, top-right,
// bottom-right, bottom-left.
using Quad = std::array<cv::Point2f, 4>;

// Computes a polygon's area/perimeter from its ordered points, using the
// shoelace formula.
double polygon_area(const std::vector<cv::Point2f>& points);
double polygon_perimeter(const std::vector<cv::Point2f>& points);

// Resizes an image to (target_w, target_h) and records how much each
// dimension was scaled.
struct ResizeResult {
    cv::Mat image;
    float ratio_h;  // target_h / original image height
    float ratio_w;  // target_w / original image width
};
ResizeResult det_resize_for_test(const cv::Mat& img, int target_h, int target_w);

// Turns the detector's raw probability map into candidate text-box quads.
class DBPostProcess {
public:
    DBPostProcess(float thresh, float box_thresh, int max_candidates, float unclip_ratio);

    // Computes the smallest rotated rectangle enclosing a set of points,
    // returned as 4 ordered corners plus its short side length ("sside").
    static std::pair<Quad, float> get_mini_boxes(const std::vector<cv::Point2f>& points);

    // Expands a quad outward by a distance based on its own area and
    // perimeter; may return more than 4 points.
    std::vector<cv::Point2f> unclip(const std::vector<cv::Point2f>& box) const;

    // Computes the average probability-map value inside a box's shape.
    static float box_score_fast(const cv::Mat& pred_map, const Quad& box);

    // Finds contours in the binary mask and turns each into a scored,
    // expanded, resized quadrilateral.
    std::vector<Quad> boxes_from_bitmap(const cv::Mat& pred_map, const cv::Mat& bitmap_u8,
                                         int dest_width, int dest_height,
                                         float ratio_w, float ratio_h) const;

    // Thresholds the probability map into a binary mask, then extracts
    // boxes from it.
    std::vector<Quad> run(const cv::Mat& pred_map, int dest_width, int dest_height,
                           float ratio_w, float ratio_h) const;

private:
    float thresh_;          // probability threshold used to binarize pred_map
    float box_thresh_;      // minimum average score for a candidate box to survive
    int max_candidates_;    // cap on contours examined per image
    float unclip_ratio_;    // how far unclip() expands each box outward
    static constexpr float kMinSize = 3.0f;
};

// Orders, clips, and filters detected quads to prepare them for cropping.
// All methods are static; this class holds no instance state.
class DetPostProcess {
public:
    static Quad order_points_clockwise(const Quad& pts);  // reorders 4 points as tl, tr, br, bl
    static Quad clip_det_res(Quad points, int img_height, int img_width);  // clamps points into the image bounds

    // Orders and clips each box, dropping any narrower or shorter than 3px.
    static std::vector<Quad> filter_tag_det_res(const std::vector<Quad>& dt_boxes,
                                                 int image_height, int image_width);
};

// Runs the full detection pipeline: resize, normalize, run the model, and
// turn its output into text-box quadrilaterals.
class TextDetector {
public:
    TextDetector(const std::string& det_model_path,
                 const std::string& target,
                 const std::string& device_id,
                 float det_thresh, float box_thresh, float unclip_ratio, int max_candidates);

    bool is_loaded() const { return model_ && model_->is_loaded(); }

    // Resizes and normalizes the image, runs the model, and converts its
    // output into quads in the original image's coordinates.
    std::vector<Quad> run(const cv::Mat& img) const;

    static constexpr int kDetH = 480;  // fixed input height the model expects
    static constexpr int kDetW = 480;  // fixed input width the model expects

private:
    std::unique_ptr<RknnExecutor> model_;  // the loaded detection model
    NormalizeImage normalize_;              // normalizes pixel values before inference
    DBPostProcess db_postprocess_;          // converts the model's raw output into boxes
};
