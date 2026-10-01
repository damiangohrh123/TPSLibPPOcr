// Compares three ways to read a drawn region with PP-OCR: recognition only, stretched detection and padded detection.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "ppocr_crop.h"
#include "ppocr_det.h"
#include "ppocr_rec.h"
#include "ppocr_system.h"
#include "raw_image.h"
#include "regions.h"

namespace {

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

enum Mode { kRec, kStretch, kPad, kModes };
const char* const kModeNames[kModes] = {"rec", "stretch", "pad"};

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
        // pad is the library's region read (tps_ppocr_read_region), so its results are the library's.
        auto read = [&](int mode) -> Reading {
            if (mode == kRec) {
                RecResult r = rec_only.run({crop})[0];
                return {r.text, r.score, {}};
            }
            if (mode == kStretch) return join_lines(system.run(crop), keep);
            return read_region(system, img, region.box, margin);
        };
        std::u32string expected = normalise(region.expected);
        for (int mode = 0; mode < kModes; ++mode) {
            Reading reading = read(mode);  // untimed warm-up run
            auto start = std::chrono::steady_clock::now();
            for (int c = 0; c < cycles; ++c) read(mode);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / cycles;
            std::string text = one_line(reading.text);
            std::u32string got = normalise(text);
            double cer = static_cast<double>(edit_distance(got, expected)) / std::max<size_t>(1, expected.size());
            bool match = got == expected;
            printf("%s\t%s\t%s\t%.1f\t%.0f\t%s\t%.2f\t%s\n", region.id.c_str(), region.category.c_str(),
                kModeNames[mode], ms, reading.score * 100, match ? "yes" : "no", cer, text.c_str());
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
