// Compares PP-OCR with the controller's Tesseract on the drawn regions of test screens, reading each region as the controller does.
// recc_build builds it with the controller's libraries (its programs/ocr_compare.cmake); tpslibppocr's own CMake does not.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "TPSConstants.h"
#include "TPSImageProcessor.h"
#include "TPSLib.h"
#include "TPSOCR.h"
#include "raw_image.h"
#include "regions.h"
#include "tps_ppocr.h"

namespace {

using TPSCommon::ImageProcessor;

// One way to read a region: PP-OCR, or Tesseract after the filters a region can set, applied in OcrManager::GetText's order.
struct Engine {
    const char* name;
    bool ppocr;
    bool raw;      // PP-OCR on the raw frame instead of the decoded JPEG: what passing raw frames to OcrManager would give
    int resize;    // the region's Resize Factor
    bool filters;  // brightness up to the maximum, sharpen 85, greyscale and remove noise: the controller's defaults for an image with no setting
};
const Engine kEngines[] = {
    {"ppocr", true, false, 1, false},
    {"ppocr_raw", true, true, 1, false},
    {"tess", false, false, 1, false},
    {"tess_x2", false, false, 2, false},
    {"tess_x3", false, false, 3, false},
    {"tess_x2_filters", false, false, 2, true},
};

const char* const kProduct = "UECC";  // the product the board's licence is for
const char* const kTessId = "ocr_compare";
const char* const kTessImage = "/tmp/ocr_compare_final.jpg";

struct Reading {
    std::string text;
    int confidence;
    std::vector<std::string> pieces;  // PP-OCR only: each detected piece as score, box corners and text
};

[[noreturn]] void fail(const std::string& what) {
    fprintf(stderr, "%s\n", what.c_str());
    std::exit(1);
}

// TPSLibOCR reports licence failures through this, at start-up and from its once-a-minute recheck.
void licence_message(const char* message, int) {
    fprintf(stderr, "%s\n", message);
}

Reading read_ppocr(tps_ppocr* handle, const unsigned char* bgr, int width, int height, const cv::Rect& box) {
    tps_ppocr_result* result = nullptr;
    if (tps_ppocr_read_region(handle, bgr, width, height, width * 3, box.x, box.y, box.width, box.height, &result)
        != TPS_PPOCR_OK) {
        fail(std::string("PP-OCR: ") + tps_ppocr_last_error());
    }
    Reading reading{result->text, result->confidence, {}};
    for (int i = 0; i < result->count; ++i) {
        const tps_ppocr_piece& p = result->pieces[i];
        char corners[128];
        snprintf(corners, sizeof corners, "%.0f,%.0f %.0f,%.0f %.0f,%.0f %.0f,%.0f",
            p.box[0], p.box[1], p.box[2], p.box[3], p.box[4], p.box[5], p.box[6], p.box[7]);
        reading.pieces.push_back(std::to_string(p.score).substr(0, 5) + "\t" + corners + "\t" + p.text);
    }
    tps_ppocr_free(result);
    return reading;
}

// OcrManager's PP-OCR path: decode the JPEG frame, then read the region.
Reading read_ppocr_jpeg(tps_ppocr* handle, const std::vector<unsigned char>& jpeg, const cv::Rect& box) {
    std::string err;
    std::vector<unsigned char> bgr;
    int width = 0, height = 0;
    if (!ImageProcessor::JpegToRaw(err, jpeg, width, height, bgr)) fail("JPEG decode: " + err);
    return read_ppocr(handle, bgr.data(), width, height, box);
}

// OcrManager's Tesseract path: crop and filter the JPEG frame, write the final image, then read that file.
Reading read_tesseract(const Engine& engine, const std::vector<unsigned char>& jpeg, const cv::Rect& box) {
    std::string err;
    std::vector<unsigned char> image = ImageProcessor::Crop(err, jpeg, box.x, box.y, box.width, box.height);
    auto check = [&](const char* stage) {
        if (image.empty()) fail(std::string("Tesseract ") + stage + ": " + err);
    };
    check("crop");
    int maxbrightness = engine.filters ? ImageProcessor::GetMaxPixelValue(err, image) : 0;
    if (engine.resize > 1) {
        image = ImageProcessor::Resize(err, image, engine.resize);
        check("resize");
    }
    if (engine.filters) {
        if (maxbrightness < 255) {
            image = ImageProcessor::Brightness(err, image, 255 - maxbrightness);
            check("brightness");
        }
        image = ImageProcessor::Sharpen(err, image, 0.85);
        check("sharpen");
        image = ImageProcessor::SetGreyscale(err, image);
        check("greyscale");
        image = ImageProcessor::RemoveNoise(err, image);
        check("remove noise");
    }
    std::ofstream(kTessImage, std::ios::binary).write(reinterpret_cast<const char*>(image.data()), image.size());
    char text[SIZE_VAL], msg[SIZE_MSG_ERR];
    int confidence = getText(kTessImage, text, msg, kTessId);
    return {confidence > -1 ? text : "", confidence, {}};
}

// The text on one line, as the regions file writes it: trimmed as OcrManager trims, line breaks as " | ", other whitespace as spaces.
std::string tidy(std::string text) {
    for (char& c : text) {
        if (c == '\r' || c == '\f' || c == '\v' || c == '\t') c = ' ';
    }
    size_t first = text.find_first_not_of(" \n");
    if (first == std::string::npos) return "";
    return one_line(text.substr(first, text.find_last_not_of(" \n") - first + 1));
}

}  // namespace

int main(int argc, char** argv) {
    const int cycles = argc > 4 ? std::atoi(argv[4]) : 0;
    if (argc < 7 || argc % 2 == 0 || cycles < 1) {
        fprintf(stderr, "usage: %s <model_dir> <tessdata_dir> <licence_file> <cycles> <screen_WxH.bgr888> <regions.tsv> "
            "[<screen_WxH.bgr888> <regions.tsv> ...]\n", argv[0]);
        return 1;
    }
    // TPSLibOCR reads only after its licence check, which needs the product set first: app's start-up order.
    setP(kProduct);
    initialiseOcr(argv[3], licence_message);
    char msg[SIZE_MSG_ERR];
    if (!initOcr(argv[2], "eng", msg, kTessId)) fail(std::string("Tesseract: ") + msg);
    tps_ppocr* handle = nullptr;
    if (tps_ppocr_create(argv[1], &handle) != TPS_PPOCR_OK) fail(std::string("PP-OCR: ") + tps_ppocr_last_error());

    printf("%d timed reads per region and engine, after one untimed read\n\n", cycles);
    printf("screen\tid\tcategory\tengine\tms\tconfidence\tmatch\tcer\ttext\n");
    std::map<std::string, Tally> tallies;
    auto count = [&](const std::string& engine, const std::vector<std::string>& groups, bool match, double cer, double ms) {
        for (const std::string& group : groups) {
            Tally& t = tallies[engine + "\t" + group];
            ++t.n;
            t.matched += match;
            t.cer += cer;
            t.ms += ms;
        }
    };
    for (int a = 5; a < argc; a += 2) {
        auto [width, height] = parse_raw_dimensions(argv[a]);
        std::ifstream f(argv[a], std::ios::binary);
        std::vector<unsigned char> raw(width > 0 && height > 0 ? static_cast<size_t>(width) * height * 3 : 0);
        if (raw.empty() || !f.read(reinterpret_cast<char*>(raw.data()), raw.size())) {
            fail(std::string("could not read ") + argv[a] + " as a raw ..._<width>x<height>.bgr888 frame");
        }
        // The Gen5 receiver hands OCR its frames as JPEG at quality 100, so both engines start from that.
        std::string err;
        std::vector<unsigned char> jpeg;
        if (!ImageProcessor::RawToJpeg(err, raw, width, height, 100, jpeg)) fail("JPEG encode: " + err);
        std::string screen(argv[a]);
        screen = screen.substr(screen.rfind('/') + 1);
        screen = screen.substr(0, screen.rfind('.'));
        std::vector<Region> regions = load_regions(argv[a + 1]);
        if (regions.empty()) fail(std::string("no regions read from ") + argv[a + 1]);

        for (const Region& region : regions) {
            const std::vector<std::string> groups = {"all", "screen " + screen, "category " + region.category};
            std::u32string expected = normalise(region.expected);
            bool best_match = false;
            double best_cer = 1e9;
            for (const Engine& engine : kEngines) {
                auto read = [&] {
                    if (!engine.ppocr) return read_tesseract(engine, jpeg, region.box);
                    return engine.raw ? read_ppocr(handle, raw.data(), width, height, region.box)
                                      : read_ppocr_jpeg(handle, jpeg, region.box);
                };
                Reading reading = read();  // untimed warm-up run
                auto start = std::chrono::steady_clock::now();
                for (int c = 0; c < cycles; ++c) read();
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / cycles;
                std::string text = tidy(reading.text);
                std::u32string got = normalise(text);
                double cer = static_cast<double>(edit_distance(got, expected)) / std::max<size_t>(1, expected.size());
                bool match = got == expected;
                printf("%s\t%s\t%s\t%s\t%.1f\t%d\t%s\t%.2f\t%s\n", screen.c_str(), region.id.c_str(), region.category.c_str(),
                    engine.name, ms, reading.confidence, match ? "yes" : "no", cer, text.c_str());
                for (const std::string& piece : reading.pieces) {
                    printf("piece\t%s\t%s\t%s\t%s\n", screen.c_str(), region.id.c_str(), engine.name, piece.c_str());
                }
                count(engine.name, groups, match, cer, ms);
                if (!engine.ppocr && cer < best_cer) {
                    best_cer = cer;
                    best_match = match;
                }
            }
            // Tesseract's best setup for this region, chosen after the fact: an upper bound no deployment can reach.
            count("tess_best", groups, best_match, best_cer, 0.0);
        }
    }

    printf("\nengine\tgroup\tmatched\tmean_cer\tmean_ms\n");
    for (const auto& [key, t] : tallies) {
        printf("%s\t%d/%d\t%.3f\t%.1f\n", key.c_str(), t.matched, t.n, t.cer / t.n, t.ms / t.n);
    }
    tps_ppocr_destroy(handle);
    disposeAll();  // also stops the licence check's thread
    return 0;
}
