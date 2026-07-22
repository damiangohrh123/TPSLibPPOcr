#include "screens.h"
#include <cstdio>
#include <stdexcept>
#include "json_value.h"

namespace {

std::vector<DetectedBox> boxes_from_ocr_response(const JsonValue& doc) {
    std::vector<DetectedBox> boxes;
    const JsonValue* boxes_val = doc.find("boxes");
    if (!boxes_val) return boxes;
    for (const auto& item : boxes_val->arr) {
        DetectedBox b;
        if (auto* t = item.find("text")) b.text = t->str;
        if (auto* sc = item.find("score")) b.score = sc->num;
        if (auto* bx = item.find("box")) {
            for (size_t i = 0; i < 4 && i < bx->arr.size(); ++i) {
                const auto& pt = bx->arr[i];
                if (pt.arr.size() >= 2) {
                    b.box[i] = {static_cast<int>(pt.arr[0].num), static_cast<int>(pt.arr[1].num)};
                }
            }
        }
        boxes.push_back(std::move(b));
    }
    return boxes;
}

}  // namespace

std::vector<std::pair<std::string, std::vector<DetectedBox>>> load_ocr_output(
    const std::string& ocr_output_dir) {
    std::vector<std::pair<std::string, std::vector<DetectedBox>>> screens;
    for (const auto& path : list_json_files(ocr_output_dir)) {
        // A malformed or empty OCR output file (e.g. a failed curl/wget
        // left a 0-byte file behind) should not take down the whole
        // program -- skip it and keep going with the rest.
        try {
            JsonValue doc = json_parse_file(path.string());
            screens.emplace_back(path.stem().string(), boxes_from_ocr_response(doc));
        } catch (const std::exception& e) {
            fprintf(stderr, "Skipping %s: %s\n", path.string().c_str(), e.what());
        }
    }
    return screens;
}
