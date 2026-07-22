#include "rule.h"
#include <cstdio>
#include <stdexcept>
#include "json_value.h"

std::vector<Step> load_rule(const std::string& path) {
    JsonValue doc = json_parse_file(path);
    std::vector<Step> rule;
    for (const auto& item : doc.arr) {
        Step s;
        if (auto* kw = item.find("keyword")) s.keyword = kw->str;
        if (auto* ac = item.find("action")) s.action = ac->str;
        if (auto* val = item.find("value")) s.value = val->str;
        rule.push_back(std::move(s));
    }
    return rule;
}

std::vector<std::pair<std::string, std::vector<Step>>> load_rules(const std::string& rules_dir) {
    std::vector<std::pair<std::string, std::vector<Step>>> rules;
    for (const auto& path : list_json_files(rules_dir)) {
        // A malformed or empty rule file (e.g. a failed download left a
        // 0-byte file behind) should not take down the whole program --
        // skip it and keep going with whatever else is in rules_dir.
        try {
            rules.emplace_back(path.stem().string(), load_rule(path.string()));
        } catch (const std::exception& e) {
            fprintf(stderr, "Skipping %s: %s\n", path.string().c_str(), e.what());
        }
    }
    return rules;
}
