// Compares three ways to read a drawn region with PP-OCR: recognition only, stretched detection and padded detection.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ppocr_det.h"
#include "ppocr_rec.h"
#include "ppocr_system.h"
#include "raw_image.h"

namespace {

struct Region {
    std::string id, category, expected;
    cv::Rect box;
};

// One region per line, tab-separated: id, category, x, y, width, height, expected text (lines joined by " | ").
std::vector<Region> load_regions(const std::string& path) {
    std::vector<Region> regions;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, '\t')) fields.push_back(field);
        if (fields.size() != 7) {
            fprintf(stderr, "skipping malformed line: %s\n", line.c_str());
            continue;
        }
        cv::Rect box(std::atoi(fields[2].c_str()), std::atoi(fields[3].c_str()),
            std::atoi(fields[4].c_str()), std::atoi(fields[5].c_str()));
        regions.push_back({fields[0], fields[1], fields[6], box});
    }
    return regions;
}

// Decodes UTF-8 to code points, dropping spaces and the " | " line separator, since neither changes a field's value.
std::u32string normalise(const std::string& s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        char32_t cp = len == 1 ? c : (c & (0xFF >> (len + 1)));
        for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
        i += len;
        if (cp != ' ' && cp != '\t' && cp != '|') out.push_back(cp);
    }
    return out;
}

size_t edit_distance(const std::u32string& a, const std::u32string& b) {
    std::vector<size_t> row(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        size_t diag = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            size_t up = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diag + (a[i - 1] != b[j - 1] ? 1 : 0)});
            diag = up;
        }
    }
    return row[b.size()];
}

struct Reading {
    std::string text;
    double score = 0.0;
};

// Joins results (already in reading order) whose box centre lies in `keep`; a centre more than half a box height from the previous one starts a new line.
Reading join(const std::vector<OcrResult>& results, const cv::Rect& keep) {
    Reading reading;
    float line_y = 0.0f, line_h = 0.0f;
    int n = 0;
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
        if (n > 0) reading.text += std::abs(centre.y - line_y) < line_h / 2 ? " " : " | ";
        line_y = centre.y;
        line_h = bottom - top;
        reading.text += res.rec.text;
        reading.score += res.rec.score;
        ++n;
    }
    if (n > 0) reading.score /= n;
    return reading;
}

// Extends the crop to at least the detector's input size with its mean edge colour, so text keeps its native scale.
cv::Mat pad_to_det_input(const cv::Mat& crop) {
    int w = crop.cols, h = crop.rows;
    cv::Scalar edge = (cv::mean(crop.row(0)) + cv::mean(crop.row(h - 1))) * static_cast<double>(w)
        + (cv::mean(crop.col(0)) + cv::mean(crop.col(w - 1))) * static_cast<double>(h);
    edge = edge * (1.0 / (2.0 * (w + h)));
    cv::Mat canvas;
    cv::copyMakeBorder(crop, canvas, 0, std::max(0, TextDetector::kDetH - h),
        0, std::max(0, TextDetector::kDetW - w), cv::BORDER_CONSTANT, edge);
    return canvas;
}

enum Mode { kRec, kStretch, kPad, kModes };
const char* const kModeNames[kModes] = {"rec", "stretch", "pad"};

Reading read_region(int mode, const cv::Mat& crop, const cv::Rect& keep,
    const TextSystem& system, const TextRecognizer& recognizer) {
    if (mode == kRec) {
        RecResult r = recognizer.run({crop})[0];
        return {r.text, r.score};
    }
    return join(system.run(mode == kPad ? pad_to_det_input(crop) : crop), keep);
}

struct Tally {
    int n = 0, matched = 0;
    double cer = 0.0, ms = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <det_model.rknn> <rec_model.rknn> <char_dict.txt> <image_WxH.bgr888> "
            "<regions.tsv> [cycles=5] [margin=0]\n", argv[0]);
        return 1;
    }
    const int cycles = argc > 6 ? std::atoi(argv[6]) : 5;
    const int margin = argc > 7 ? std::atoi(argv[7]) : 0;

    cv::Mat img = load_image(argv[4]);
    if (img.empty()) return 1;
    std::vector<Region> regions = load_regions(argv[5]);
    if (regions.empty()) {
        fprintf(stderr, "no regions read from %s\n", argv[5]);
        return 1;
    }

    TextDetector detector(argv[1], 0.2f, 0.4f, 1.5f, 3000);
    TextRecognizer recognizer(argv[2], argv[3]);
    TextRecognizer rec_only(argv[2], argv[3]);
    if (!detector.is_loaded() || !recognizer.is_loaded() || !rec_only.is_loaded()) {
        fprintf(stderr, "a model failed to load\n");
        return 1;
    }
    TextSystem system(std::move(detector), std::move(recognizer), 0.4, 10.0, 8.0);

    printf("image %s, %zu regions, %d timed cycles per mode, margin %d px\n\n",
        argv[4], regions.size(), cycles, margin);
    printf("id\tcategory\tmode\tms\tscore\tmatch\tcer\ttext\n");
    std::map<std::string, Tally> tallies;
    for (const auto& region : regions) {
        cv::Rect outer = cv::Rect(region.box.x - margin, region.box.y - margin,
            region.box.width + 2 * margin, region.box.height + 2 * margin) & cv::Rect(0, 0, img.cols, img.rows);
        cv::Mat crop = img(outer).clone();
        cv::Rect keep(region.box.x - outer.x, region.box.y - outer.y, region.box.width, region.box.height);
        std::u32string expected = normalise(region.expected);
        for (int mode = 0; mode < kModes; ++mode) {
            Reading reading = read_region(mode, crop, keep, system, rec_only);  // untimed warm-up run
            auto start = std::chrono::steady_clock::now();
            for (int c = 0; c < cycles; ++c) read_region(mode, crop, keep, system, rec_only);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / cycles;
            std::u32string got = normalise(reading.text);
            double cer = static_cast<double>(edit_distance(got, expected)) / std::max<size_t>(1, expected.size());
            bool match = got == expected;
            printf("%s\t%s\t%s\t%.1f\t%.0f\t%s\t%.2f\t%s\n", region.id.c_str(), region.category.c_str(),
                kModeNames[mode], ms, reading.score * 100, match ? "yes" : "no", cer, reading.text.c_str());
            for (const std::string& group : {region.category, std::string("all")}) {
                Tally& t = tallies[std::string(kModeNames[mode]) + "\t" + group];
                ++t.n;
                t.matched += match;
                t.cer += cer;
                t.ms += ms;
            }
        }
    }

    printf("\nmode\tcategory\tmatched\tmean_cer\tmean_ms\n");
    for (const auto& [key, t] : tallies) {
        printf("%s\t%d/%d\t%.3f\t%.1f\n", key.c_str(), t.matched, t.n, t.cer / t.n, t.ms / t.n);
    }
    return 0;
}
