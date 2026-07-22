#include "text_correction.h"
#include <cctype>
#include <vector>

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_alnum(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }

// Rewrites standalone digit-dash-digit tokens (e.g. "0-000", "203-0") into
// a decimal number, since PP-OCRv6 recurringly misreads "." as "-" on these
// screens. Only touches a token that is *entirely* digits with exactly one
// dash and clean token boundaries on both sides -- multi-dash tokens like
// device IDs ("ADAU1787-AD3-SCDD-CB-HBEA") or anything containing letters
// is left untouched, since the misread only ever shows up on plain numeric
// fields in real testing.
std::string fix_decimal_dash(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        bool boundary_before = (start == 0) || !is_alnum(text[start - 1]);
        size_t j = start;
        while (j < text.size() && is_digit(text[j])) ++j;

        if (j > start && boundary_before && j < text.size() && text[j] == '-') {
            size_t dash = j;
            size_t k = dash + 1;
            while (k < text.size() && is_digit(text[k])) ++k;
            bool digits_after = k > dash + 1;
            bool boundary_after = (k == text.size()) || !is_alnum(text[k]);

            if (digits_after && boundary_after) {
                result += text.substr(start, dash - start);
                result += '.';
                result += text.substr(dash + 1, k - dash - 1);
                i = k;
                continue;
            }
        }
        result += text[i];
        ++i;
    }
    return result;
}

// Small, fixed vocabulary of this project's actual recurring UI labels.
// Used only to snap a recognized word that differs from one of these by a
// single l/I/1 substitution back to the known-correct spelling. Narrow by
// design: these are fixed HMI screens with a small, known vocabulary, not
// open-domain text -- a short list like this is a reasonable fix for a
// documented, recurring confusion pattern, not a general spell-checker.
// Extend this list as new real recurring labels are confirmed.
const std::vector<std::string>& known_vocabulary() {
    static const std::vector<std::string> kWords = {
        "Full", "Automation", "Channel", "line", "lines", "slot",
        "Ch1", "Ch2", "Ch3", "Ch4",
    };
    return kWords;
}

bool is_confusable(char c) { return c == 'l' || c == 'I' || c == '1'; }

// True if a and b are identical except at positions where both characters
// are drawn from the confusable set {'l', 'I', '1'}.
bool matches_ignoring_l_I_1(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) continue;
        if (is_confusable(a[i]) && is_confusable(b[i])) continue;
        return false;
    }
    return true;
}

// Tokenizes on whitespace, snapping any token that matches a known
// vocabulary word (ignoring l/I/1 substitutions) to that word's correct
// spelling.
std::string fix_l_I_1_confusion(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        std::string token = text.substr(start, i - start);

        for (const auto& known : known_vocabulary()) {
            if (matches_ignoring_l_I_1(token, known)) {
                token = known;
                break;
            }
        }
        result += token;
        if (i < text.size()) result += text[i++];  // preserve the whitespace character
    }
    return result;
}

}  // namespace

std::string correct_known_ocr_errors(const std::string& text) {
    return fix_l_I_1_confusion(fix_decimal_dash(text));
}
