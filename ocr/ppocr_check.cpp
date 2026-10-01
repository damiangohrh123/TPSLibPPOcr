// Loads libTPSLibPPOcr.so as the controller will, then reads a raw .bgr888 frame whole and, given a regions file, region by region.
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "raw_image.h"
#include "regions.h"
#include "tps_ppocr.h"

namespace {

// Looks up one of the library's functions, exiting if it is missing.
template <typename F>
F symbol(void* lib, const char* name) {
    F f = reinterpret_cast<F>(dlsym(lib, name));
    if (!f) {
        fprintf(stderr, "%s\n", dlerror());
        std::exit(1);
    }
    return f;
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <libTPSLibPPOcr.so> <model_dir> <image_WxH.bgr888> [regions.tsv]\n", argv[0]);
        return 1;
    }
    // RTLD_NOW, as the controller loads its libraries: every function the library calls must resolve now.
    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    const auto abi_version = symbol<decltype(&tps_ppocr_abi_version)>(lib, "tps_ppocr_abi_version");
    const auto create = symbol<decltype(&tps_ppocr_create)>(lib, "tps_ppocr_create");
    const auto read_frame = symbol<decltype(&tps_ppocr_read_frame)>(lib, "tps_ppocr_read_frame");
    const auto read_region = symbol<decltype(&tps_ppocr_read_region)>(lib, "tps_ppocr_read_region");
    const auto free_result = symbol<decltype(&tps_ppocr_free)>(lib, "tps_ppocr_free");
    const auto destroy = symbol<decltype(&tps_ppocr_destroy)>(lib, "tps_ppocr_destroy");
    const auto last_error = symbol<decltype(&tps_ppocr_last_error)>(lib, "tps_ppocr_last_error");
    printf("library ABI %d, header ABI %d\n", abi_version(), TPS_PPOCR_ABI_VERSION);

    auto [width, height] = parse_raw_dimensions(argv[3]);
    std::ifstream f(argv[3], std::ios::binary);
    std::vector<unsigned char> frame(width > 0 && height > 0 ? static_cast<size_t>(width) * height * 3 : 0);
    if (frame.empty() || !f.read(reinterpret_cast<char*>(frame.data()), frame.size())) {
        fprintf(stderr, "could not read %s as a raw ..._<width>x<height>.bgr888 frame\n", argv[3]);
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    tps_ppocr* ocr = nullptr;
    if (create(argv[2], &ocr) != TPS_PPOCR_OK) {
        fprintf(stderr, "tps_ppocr_create: %s\n", last_error());
        return 1;
    }
    printf("models loaded in %.0f ms\n\n", ms_since(start));

    start = std::chrono::steady_clock::now();
    tps_ppocr_result* result = nullptr;
    if (read_frame(ocr, frame.data(), width, height, width * 3, &result) != TPS_PPOCR_OK) {
        fprintf(stderr, "tps_ppocr_read_frame: %s\n", last_error());
        return 1;
    }
    printf("frame %dx%d: %d pieces in %.0f ms, confidence %d\n", width, height, result->count, ms_since(start), result->confidence);
    printf("score\tbox\ttext\n");
    for (int i = 0; i < result->count; ++i) {
        const tps_ppocr_piece& p = result->pieces[i];
        printf("%.3f\t%.0f,%.0f %.0f,%.0f %.0f,%.0f %.0f,%.0f\t%s\n", p.score, p.box[0], p.box[1], p.box[2], p.box[3],
            p.box[4], p.box[5], p.box[6], p.box[7], p.text);
    }
    free_result(result);

    if (argc > 4) {
        std::vector<Region> regions = load_regions(argv[4]);
        if (regions.empty()) {
            fprintf(stderr, "no regions read from %s\n", argv[4]);
            return 1;
        }
        printf("\nid\tms\tconfidence\tmatch\ttext\n");
        int matched = 0;
        double total_ms = 0.0;
        for (const auto& region : regions) {
            start = std::chrono::steady_clock::now();
            if (read_region(ocr, frame.data(), width, height, width * 3, region.box.x, region.box.y,
                    region.box.width, region.box.height, &result) != TPS_PPOCR_OK) {
                fprintf(stderr, "tps_ppocr_read_region %s: %s\n", region.id.c_str(), last_error());
                return 1;
            }
            double ms = ms_since(start);
            total_ms += ms;
            std::string text = one_line(result->text);
            bool match = normalise(text) == normalise(region.expected);
            matched += match;
            printf("%s\t%.1f\t%d\t%s\t%s\n", region.id.c_str(), ms, result->confidence, match ? "yes" : "no", text.c_str());
            free_result(result);
        }
        printf("\n%d/%zu regions read exactly, %.1f ms per region\n", matched, regions.size(), total_ms / regions.size());
    }
    destroy(ocr);
    return 0;
}
