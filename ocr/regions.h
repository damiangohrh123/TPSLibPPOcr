#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// One region drawn on a test screen, with the text it shows.
struct Region {
    std::string id, category, expected;
    cv::Rect box;
};

// One region per line, tab-separated: id, category, x, y, width, height, expected text (lines joined by " | ").
inline std::vector<Region> load_regions(const std::string& path) {
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

// Replaces line breaks with " | ", the regions file's line separator.
inline std::string one_line(std::string text) {
    for (size_t i = text.find('\n'); i != std::string::npos; i = text.find('\n', i + 3)) text.replace(i, 1, " | ");
    return text;
}

// Decodes UTF-8 to code points, dropping spaces and the " | " line separator, since neither changes a field's value.
inline std::u32string normalise(const std::string& s) {
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
