#include "ppocr_crop.h"
#include <algorithm>
#include <cmath>

Reading join_lines(const std::vector<OcrResult>& results, const cv::Rect& keep) {
    Reading reading;
    float line_y = 0.0f, line_h = 0.0f;
    for (const auto& res : results) {
        cv::Point2f centre(0.0f, 0.0f);
        float top = res.box[0].y, bottom = res.box[0].y;
        for (const auto& p : res.box) {
            centre += p;
            top = std::min(top, p.y);
            bottom = std::max(bottom, p.y);
        }
        centre *= 0.25f;
        if (!keep.contains(cv::Point(cvRound(centre.x), cvRound(centre.y)))) continue;
        if (!reading.pieces.empty()) reading.text += std::abs(centre.y - line_y) < line_h / 2 ? " " : "\n";
        line_y = centre.y;
        line_h = bottom - top;
        reading.text += res.rec.text;
        reading.score += res.rec.score;
        reading.pieces.push_back(res);
    }
    if (!reading.pieces.empty()) reading.score /= reading.pieces.size();
    return reading;
}

cv::Mat pad_to_det_input(const cv::Mat& crop) {
    int w = crop.cols, h = crop.rows;
    cv::Scalar edge = (cv::mean(crop.row(0)) + cv::mean(crop.row(h - 1))) * static_cast<double>(w)
        + (cv::mean(crop.col(0)) + cv::mean(crop.col(w - 1))) * static_cast<double>(h);
    edge = edge * (1.0 / (2.0 * (w + h)));
    cv::Mat canvas;
    // BORDER_ISOLATED: a crop taken from a frame is filled with the edge colour, not with the frame's own pixels.
    cv::copyMakeBorder(crop, canvas, 0, std::max(0, TextDetector::kDetH - h),
        0, std::max(0, TextDetector::kDetW - w), cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, edge);
    return canvas;
}

Reading read_region(const TextSystem& system, const cv::Mat& frame, const cv::Rect& region, int margin) {
    cv::Rect outer = cv::Rect(region.x - margin, region.y - margin, region.width + 2 * margin, region.height + 2 * margin)
        & cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Rect keep(region.x - outer.x, region.y - outer.y, region.width, region.height);
    Reading reading = join_lines(system.run(pad_to_det_input(frame(outer))), keep);
    for (auto& piece : reading.pieces) {
        for (auto& p : piece.box) p += cv::Point2f(static_cast<float>(outer.x), static_cast<float>(outer.y));
    }
    return reading;
}
