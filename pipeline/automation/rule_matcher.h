#pragma once
#include <array>
#include <string>
#include <vector>

// Matches an offline-drafted, pre-tested rule against real OCR output,
// and turns a match into a KVM action sequence. Deliberately has zero
// OpenCV/RKNN dependency -- this logic never touches an image, only the
// text and coordinates OCR already produced, so it builds and runs
// anywhere, unlike the OCR pipeline itself.

// One detected text box, same shape as a real POST /ocr response's
// "boxes" entries: text, confidence score, and 4 corner points
// (top-left, top-right, bottom-right, bottom-left).
struct DetectedBox {
    std::string text;
    double score = 0.0;
    std::array<std::array<int, 2>, 4> box{};
};

// One step of an offline-drafted rule: a keyword to look for, and what
// to do once it's found. Written ahead of time (by a person, or a real
// LLM whose output a person tests before trusting) -- only the matching
// below runs live.
struct Step {
    std::string keyword;
    std::string action;  // "type" or "click"
    std::string value;   // only used when action == "type"
};

// One concrete KVM action.
struct Action {
    std::string type;  // "move", "click", or "type"
    int x = 0;
    int y = 0;
    std::string text;  // only used when type == "type"
};

struct MatchResult {
    std::vector<Action> actions;
    bool complete = false;  // false = some step couldn't be confidently
                             // matched; stop and let a person check
};

// Ratcliff/Obershelp similarity ratio between two strings, in [0, 1].
// Same algorithm Python's difflib.SequenceMatcher.ratio() uses (minus
// the "autojunk" heuristic, which only applies to sequences of 200+
// elements and is irrelevant for short UI labels).
double sequence_ratio(const std::string& a, const std::string& b);

// Returns the box whose text best matches the keyword, and its
// confidence score. Returns {nullptr, 0.0} if boxes is empty.
std::pair<const DetectedBox*, double> find_by_keyword(const std::string& keyword,
                                                       const std::vector<DetectedBox>& boxes);

// Center point of a 4-corner box.
std::pair<int, int> box_center(const std::array<std::array<int, 2>, 4>& box);

// Guesses an empty input field's position from its label, by checking
// for open space immediately to the right of the label, space no other
// detected box already occupies.
std::pair<int, int> find_field_near_label(const DetectedBox& label_box,
                                           const std::vector<DetectedBox>& boxes);

// Matches every step of a rule against the given screen's boxes, in
// order. Below CONFIDENCE_THRESHOLD, stops immediately instead of
// guessing (result.complete = false).
MatchResult match_rule_to_screen(const std::vector<Step>& rule,
                                  const std::vector<DetectedBox>& boxes,
                                  double confidence_threshold);
