#pragma once
#include <string>
#include <utility>
#include <vector>
#include "rule_matcher.h"

// A rule is a JSON file a person writes by hand (or reviews and tests
// before trusting, if something else drafted it first) and drops into
// the rules/ directory. No code change or rebuild is needed to add one.
//
// Shape: an array of steps, each either a click:
//   {"keyword": "OK", "action": "click"}
// or a type-then-implicit-enter:
//   {"keyword": "Power", "action": "type", "value": "75"}
//
// A full rule file, e.g. rules/acknowledge_alarm.json:
//   [
//     {"keyword": "Acknowledge", "action": "click"}
//   ]

std::vector<Step> load_rule(const std::string& path);

// Loads every rules/*.json file into a (rule name, steps) pair, sorted
// by filename. Returns an empty vector if the directory doesn't exist
// or has no .json files in it.
std::vector<std::pair<std::string, std::vector<Step>>> load_rules(const std::string& rules_dir);
