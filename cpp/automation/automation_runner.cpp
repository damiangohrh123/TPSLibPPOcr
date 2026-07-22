// Goal-based automation runner: loads every rule in rules/ and every
// real OCR result in ocr_output/, matches each rule against each
// screen, and prints the resulting KVM action sequence.
//
// A rule is a small JSON file a person has written and tested against
// the real screen (see rule.h for the exact shape); adding one is a
// file drop, not a code change. A screen is a real POST /ocr response
// saved from the actual OCR server (see screens.h); getting one is:
//
//   curl -X POST --data-binary @testdata/alarm.jpg http://<board-ip>:8080/ocr
//
// Matching is a keyword search against whatever OCR actually read off
// the current screen, with a confidence threshold: below it, this stops
// and flags the step for a person to check instead of guessing (see
// rule_matcher.h). Nothing here is sent to a machine -- the resulting
// sequence is printed and waits for a person to approve it before it
// would ever run for real (see kvm_sequence.h).
//
// Deliberately has no OpenCV/RKNN dependency: this logic never touches
// an image, only text and coordinates OCR already produced, so it
// builds and runs on any machine, not just the RECC board.

#include <cstdio>
#include <string>
#include <vector>
#include "kvm_sequence.h"
#include "rule.h"
#include "rule_matcher.h"
#include "screens.h"

namespace {

void run_pipeline_for_screen(const std::string& name, const std::vector<Step>& rule,
                              const std::vector<DetectedBox>& boxes) {
    printf("======================================================================\n");
    printf("Matching rule against: %s\n", name.c_str());
    printf("======================================================================\n");
    MatchResult result = match_rule_to_screen(rule, boxes, /*confidence_threshold=*/0.75);
    printf("======================================================================\n");
    if (result.complete) {
        render_kvm_sequence(result.actions);
    } else {
        printf("Sequence incomplete. Nothing will run until a person reviews this step.\n");
    }
    printf("======================================================================\n\n");
}

}  // namespace

int main() {
    const std::string rules_dir = "rules";
    const std::string ocr_output_dir = "ocr_output";

    auto rules = load_rules(rules_dir);
    if (rules.empty()) {
        printf("No rules found in %s/\n\n", rules_dir.c_str());
        printf("Add one by writing a JSON file there, e.g. %s/acknowledge_alarm.json:\n",
               rules_dir.c_str());
        printf("  [{\"keyword\": \"Acknowledge\", \"action\": \"click\"}]\n");
        return 0;
    }

    auto screens = load_ocr_output(ocr_output_dir);
    if (screens.empty()) {
        printf("No real OCR output found in %s/\n\n", ocr_output_dir.c_str());
        printf("Run the actual OCR server on the RECC board against a real image, e.g.:\n");
        printf("  curl -X POST --data-binary @testdata/alarm.jpg http://<board-ip>:8080/ocr\n");
        printf("and save the response body as %s/alarm.json, then re-run this program.\n",
               ocr_output_dir.c_str());
        return 0;
    }

    for (const auto& [rule_name, steps] : rules) {
        for (const auto& [screen_name, boxes] : screens) {
            run_pipeline_for_screen(rule_name + " -- on screen: " + screen_name, steps, boxes);
        }
    }
    return 0;
}
