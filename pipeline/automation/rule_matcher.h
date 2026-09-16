#pragma once
#include <array>
#include <string>
#include <vector>

// Fuzzy-matches a rule's keyword against real OCR output and locates the box
// to act on; step_matcher.cpp is what turns that into an action. Deliberately
// has zero OpenCV/RKNN dependency -- this logic never touches an image, only
// the text and coordinates OCR already produced, so it builds and runs
// anywhere, unlike the OCR pipeline itself.

// One detected text box, same shape as a real POST /ocr response's
// "boxes" entries: text, confidence score, and 4 corner points
// (top-left, top-right, bottom-right, bottom-left).
struct DetectedBox {
    std::string text;
    double score = 0.0;
    std::array<std::array<int, 2>, 4> box{};
};

// The box a person drew when the step was authored, scaled to the live frame.
// Every step carries one; it says where on screen the step acts.
struct Region {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

// A keyword match over a region's text, with the run of words that matched.
struct KeywordMatch {
    std::string text;
    double score = 0.0;
};

// True when any part of the box falls inside the region. Overlap rather than
// containment is what absorbs the few pixels of drift between captures,
// without a padding value to tune.
bool overlaps(const DetectedBox& b, const Region& region);

// Returns the box inside the region whose text best matches the keyword, and
// its score in [0, 1]. The keyword is compared against a run of its own word
// count within each box's text, so surrounding words on the same box do not
// dilute it. Returns {nullptr, 0.0} when no box overlaps the region.
std::pair<const DetectedBox*, double> find_by_keyword(const std::string& keyword,
                                                       const std::vector<DetectedBox>& boxes,
                                                       const Region& region);

// Best match for any of the keywords against the text of every box overlapping
// the region, compared word by word so a phrase OCR split across two boxes
// still matches. Sizing each comparison to the keyword's own word count keeps a
// short keyword from being diluted by a long line of surrounding text.
KeywordMatch find_keyword_in_region(const std::vector<std::string>& keywords,
                                     const std::vector<DetectedBox>& boxes,
                                     const Region& region);

// Center point of a 4-corner box.
std::pair<int, int> box_center(const std::array<std::array<int, 2>, 4>& box);

// Center point of a region, which is where a type step aims.
std::pair<int, int> region_center(const Region& region);
