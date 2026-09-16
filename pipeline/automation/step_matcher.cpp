// Checks ONE rule step against ONE freshly captured screen and prints the
// result as a single line of JSON. Meant to be invoked once per live
// cycle by a driver that re-reads the screen between steps (see
// recc_gen5_test_kit/automation/README.md) -- a real machine's
// screen changes after every action, so live execution has to re-check
// one step at a time against a fresh read, rather than matching a whole
// rule against one saved screen in a single shot.
//
// Every step carries the box a person drew when it was authored, already
// scaled to the live frame by the driver. Only text overlapping that box is
// considered, which is what separates the intended match from an identical
// label elsewhere on the screen.
//
// Usage: step_matcher <screen.json> <action> <x> <y> <w> <h> [keyword ...]
//   click  one keyword; the best-matching box inside the region is clicked
//   halt   one or more keywords, compared word by word across the region's
//          whole text, so a phrase OCR split across two boxes still matches
//   type   no keyword; returns the region's center to type into, and the text
//          currently in it, which the driver compares against what it wrote
//
// Prints one line of JSON, e.g.:
//   {"matched":true,"confidence":0.84,"best_text":"0089 Kerf check: off center","action":"click","x":296,"y":69}
//   {"matched":false,"confidence":0.67,"best_text":"sTOAT","action":"click"}
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "api/json_write.h"
#include "json_value.h"
#include "rule_matcher.h"

namespace {

// Turns a real POST /ocr response into DetectedBoxes. Same parsing for a
// freshly captured screen and a replayed saved one.
std::vector<DetectedBox> load_screen(const std::string& path) {
    std::vector<DetectedBox> boxes;
    const JsonValue doc = json_parse_file(path);
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

// The text of every box overlapping the region, joined in reading order. A
// type step compares this against the value it wrote.
std::string region_text(const std::vector<DetectedBox>& boxes, const Region& region) {
    std::string out;
    for (const auto& b : boxes) {
        if (!overlaps(b, region)) continue;
        if (!out.empty()) out += ' ';
        out += b.text;
    }
    return out;
}

// The accept/reject decision for one step lives here: rule_matcher.cpp's
// matchers return a raw score and leave the threshold to their caller.
// automation_driver.py mirrors this value for log messages only.
constexpr double kConfidenceThreshold = 0.75;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <screen.json> <action> <x> <y> <w> <h> [keyword ...]\n", argv[0]);
        return 2;
    }
    const std::string screen_path = argv[1];
    const std::string action = argv[2];
    Region region{std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6])};
    std::vector<std::string> keywords(argv + 7, argv + argc);

    if (action != "click" && action != "type" && action != "halt") {
        fprintf(stderr, "unrecognized action %s (expected \"click\", \"type\" or \"halt\")\n", action.c_str());
        return 2;
    }
    if (action != "type" && keywords.empty()) {
        fprintf(stderr, "action %s needs at least one keyword\n", action.c_str());
        return 2;
    }

    std::vector<DetectedBox> boxes;
    try {
        boxes = load_screen(screen_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "failed to load %s: %s\n", screen_path.c_str(), e.what());
        return 1;
    }

    if (action == "type") {
        auto [cx, cy] = region_center(region);
        printf("{\"matched\":true,\"confidence\":1.0000,\"best_text\":\"%s\",\"action\":\"type\",\"x\":%d,\"y\":%d}\n",
               json_escape(region_text(boxes, region)).c_str(), cx, cy);
        return 0;
    }

    if (action == "halt") {
        KeywordMatch m = find_keyword_in_region(keywords, boxes, region);
        printf("{\"matched\":%s,\"confidence\":%.4f,\"best_text\":\"%s\",\"action\":\"halt\"}\n",
               m.score >= kConfidenceThreshold ? "true" : "false", m.score,
               json_escape(m.text).c_str());
        return 0;
    }

    auto [match, score] = find_by_keyword(keywords[0], boxes, region);
    const bool matched = (match != nullptr && score >= kConfidenceThreshold);
    printf("{\"matched\":%s,\"confidence\":%.4f,\"best_text\":\"%s\",\"action\":\"click\"",
           matched ? "true" : "false", score, json_escape(match ? match->text : "").c_str());
    if (matched) {
        auto [cx, cy] = box_center(match->box);
        printf(",\"x\":%d,\"y\":%d", cx, cy);
    }
    printf("}\n");
    return 0;
}
