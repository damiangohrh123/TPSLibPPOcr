#pragma once
#include <string>
#include <utility>
#include <vector>
#include "rule_matcher.h"

// Loads every ocr_output/*.json file (a real POST /ocr response) into a
// (screen name, boxes) pair, sorted by filename. Returns an empty vector
// if the directory doesn't exist or has no .json files in it -- callers
// should treat that as "no real OCR output yet", not an error.
std::vector<std::pair<std::string, std::vector<DetectedBox>>> load_ocr_output(
    const std::string& ocr_output_dir);
