// Entry point for the Internal OCR API server: loads the det/rec models
// once, then serves OCR + alarm-detection requests over HTTP until killed.
// See the "Internal OCR API" section of the top-level README.md for the
// request/response contract this implements.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>
#include "pipeline/ocr/alarm_detector.h"
#include "api/http_server.h"
#include "api/json_write.h"
#include "pipeline/ocr/ppocr_det.h"
#include "pipeline/ocr/ppocr_rec.h"
#include "pipeline/ocr/ppocr_system.h"
#include "pipeline/ocr/image_io.h"

namespace {

std::string box_to_json(const Quad& box) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < box.size(); ++i) {
        if (i) out << ",";
        out << "[" << box[i].x << "," << box[i].y << "]";
    }
    out << "]";
    return out.str();
}

std::string results_to_json(const std::vector<OcrResult>& results, const AlarmResult& alarm,
                             double timing_ms) {
    std::ostringstream out;
    out << "{\"boxes\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i) out << ",";
        out << "{\"text\":\"" << json_escape(r.rec.text) << "\","
            << "\"score\":" << r.rec.score << ","
            << "\"box\":" << box_to_json(r.box) << "}";
    }
    out << "],\"alarm\":{\"detected\":" << (alarm.alarm ? "true" : "false");
    if (alarm.alarm) {
        out << ",\"bbox\":[" << alarm.bbox.x << "," << alarm.bbox.y << ","
            << alarm.bbox.width << "," << alarm.bbox.height << "]";
        if (alarm.text.has_value()) {
            out << ",\"text\":\"" << json_escape(*alarm.text) << "\"";
        }
    }
    out << "}," << "\"timing_ms\":" << timing_ms << "}";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <det_model.rknn> <rec_model.rknn> <char_dict.txt> <port>\n", argv[0]);
        return 1;
    }
    const std::string det_model_path = argv[1];
    const std::string rec_model_path = argv[2];
    const std::string char_dict_path = argv[3];
    const int port = std::atoi(argv[4]);

    printf("loading models...\n");
    // Empty target/device_id runs inference on this board's own NPU, same
    // as main.cpp.
    TextDetector detector(det_model_path, /*target=*/"", /*device_id=*/"",
        /*det_thresh=*/0.3f, /*box_thresh=*/0.4f,
        /*unclip_ratio=*/1.5f, /*max_candidates=*/3000);
    if (!detector.is_loaded()) {
        fprintf(stderr, "det model failed to load\n");
        return 1;
    }

    TextRecognizer recognizer(rec_model_path, /*target=*/"", /*device_id=*/"", char_dict_path);
    if (!recognizer.is_loaded()) {
        fprintf(stderr, "rec model failed to load\n");
        return 1;
    }

    TextSystem text_system(std::move(detector), std::move(recognizer),
        /*drop_score=*/0.4, /*min_height=*/10.0, /*min_width=*/8.0);
    AlarmDetector alarm_detector;

    printf("models loaded, serving on port %d\n", port);

    HttpServer server(port);

    server.on("GET", "/health", [](const HttpRequest&) {
        HttpResponse res;
        res.body = "{\"status\":\"ok\"}";
        return res;
    });

    server.on("POST", "/ocr", [&](const HttpRequest& req) {
        HttpResponse res;
        cv::Mat img = decode_image_from_memory(
            reinterpret_cast<const unsigned char*>(req.body.data()), req.body.size());
        if (img.empty()) {
            res.status = 400;
            res.body = "{\"error\":\"could not decode image\"}";
            return res;
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<OcrResult> results = text_system.run(img);
        AlarmResult alarm = alarm_detector.detect(img, results);
        auto t1 = std::chrono::steady_clock::now();
        double timing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        res.body = results_to_json(results, alarm, timing_ms);
        return res;
    });

    server.run();
    return 0;
}
