#include "tps_ppocr.h"
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>
#include "ppocr_crop.h"
#include "ppocr_det.h"
#include "ppocr_rec.h"
#include "ppocr_system.h"

struct tps_ppocr {
    explicit tps_ppocr(TextSystem s) : system(std::move(s)) {}
    TextSystem system;
    std::mutex mutex;
};

namespace {

// The tuned settings benchmark and crop_eval use, and Appendix E's 8 pixel margin.
constexpr float kDetThresh = 0.2f, kBoxThresh = 0.4f, kUnclipRatio = 1.5f;
constexpr int kMaxCandidates = 3000;
constexpr double kDropScore = 0.4, kMinHeight = 10.0, kMinWidth = 8.0;
constexpr int kMargin = 8;

thread_local std::string last_error;

int fail(int code, const std::string& message) {
    last_error = message;
    return code;
}

// A result together with the storage its pointers refer to.
struct Result : tps_ppocr_result {
    Reading reading;
    std::vector<tps_ppocr_piece> c_pieces;
};

tps_ppocr_result* to_result(Reading reading) {
    auto r = std::make_unique<Result>();
    r->reading = std::move(reading);
    r->c_pieces.reserve(r->reading.pieces.size());
    for (const auto& piece : r->reading.pieces) {
        tps_ppocr_piece p;
        p.text = piece.rec.text.c_str();
        p.score = piece.rec.score;
        for (int i = 0; i < 4; ++i) {
            p.box[2 * i] = piece.box[i].x;
            p.box[2 * i + 1] = piece.box[i].y;
        }
        r->c_pieces.push_back(p);
    }
    r->text = r->reading.text.c_str();
    r->confidence = static_cast<int>(std::lround(r->reading.score * 100));
    r->count = static_cast<int>(r->c_pieces.size());
    r->pieces = r->c_pieces.data();
    return r.release();
}

// Checks the frame, then reads the region, or the whole frame when region is null, under the handle's lock.
int read_checked(tps_ppocr* handle, const unsigned char* bgr, int width, int height, int stride, const cv::Rect* region,
    tps_ppocr_result** out_result) {
    if (!out_result) return fail(TPS_PPOCR_ERR_INVALID_ARG, "out_result is null");
    *out_result = nullptr;
    if (!handle || !bgr) return fail(TPS_PPOCR_ERR_INVALID_ARG, "handle or bgr is null");
    if (width <= 0 || height <= 0 || stride < 3 * width) {
        return fail(TPS_PPOCR_ERR_INVALID_ARG, "bad frame size " + std::to_string(width) + "x" + std::to_string(height)
            + ", stride " + std::to_string(stride));
    }
    const cv::Rect whole(0, 0, width, height);
    if (region && (region->empty() || (*region & whole) != *region)) {
        return fail(TPS_PPOCR_ERR_INVALID_ARG, "region " + std::to_string(region->x) + "," + std::to_string(region->y)
            + " " + std::to_string(region->width) + "x" + std::to_string(region->height) + " is not inside the frame");
    }
    try {
        cv::Mat frame(height, width, CV_8UC3, const_cast<unsigned char*>(bgr), static_cast<size_t>(stride));
        std::lock_guard<std::mutex> lock(handle->mutex);
        *out_result = to_result(region ? read_region(handle->system, frame, *region, kMargin)
                                       : join_lines(handle->system.run(frame), whole));
    } catch (const std::exception& e) {
        return fail(TPS_PPOCR_ERR_READ_FAILED, e.what());
    }
    return TPS_PPOCR_OK;
}

}  // namespace

extern "C" {

int tps_ppocr_abi_version(void) {
    return TPS_PPOCR_ABI_VERSION;
}

int tps_ppocr_create(const char* model_dir, tps_ppocr** out_handle) {
    if (!out_handle) return fail(TPS_PPOCR_ERR_INVALID_ARG, "out_handle is null");
    *out_handle = nullptr;
    if (!model_dir) return fail(TPS_PPOCR_ERR_INVALID_ARG, "model_dir is null");
    try {
        const std::string dir(model_dir);
        TextDetector detector(dir + "/PP-OCRv6_tiny_det_rk3588.rknn", kDetThresh, kBoxThresh, kUnclipRatio, kMaxCandidates);
        TextRecognizer recognizer(dir + "/PP-OCRv6_tiny_rec_rk3588.rknn", dir + "/ppocr_keys_v6.txt");
        if (!detector.is_loaded() || !recognizer.is_loaded()) {
            return fail(TPS_PPOCR_ERR_LOAD_FAILED, "could not load the models from " + dir + "; stderr has the details");
        }
        *out_handle = new tps_ppocr(TextSystem(std::move(detector), std::move(recognizer), kDropScore, kMinHeight, kMinWidth));
    } catch (const std::exception& e) {
        return fail(TPS_PPOCR_ERR_LOAD_FAILED, e.what());
    }
    return TPS_PPOCR_OK;
}

int tps_ppocr_read_frame(tps_ppocr* handle, const unsigned char* bgr, int width, int height, int stride,
    tps_ppocr_result** out_result) {
    return read_checked(handle, bgr, width, height, stride, nullptr, out_result);
}

int tps_ppocr_read_region(tps_ppocr* handle, const unsigned char* bgr, int width, int height, int stride,
    int x, int y, int region_width, int region_height, tps_ppocr_result** out_result) {
    const cv::Rect region(x, y, region_width, region_height);
    return read_checked(handle, bgr, width, height, stride, &region, out_result);
}

void tps_ppocr_free(tps_ppocr_result* result) {
    delete static_cast<Result*>(result);
}

void tps_ppocr_destroy(tps_ppocr* handle) {
    delete handle;
}

const char* tps_ppocr_last_error(void) {
    return last_error.c_str();
}

}  // extern "C"
