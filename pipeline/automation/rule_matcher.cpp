#include "rule_matcher.h"
#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    return out;
}

struct Match {
    int a = 0;
    int b = 0;
    int size = 0;
};

// Longest contiguous run common to a[alo,ahi) and b[blo,bhi). Same
// dynamic-programming approach difflib.SequenceMatcher.find_longest_match
// uses internally.
Match find_longest_match(const std::string& a, int alo, int ahi,
                          const std::string& b, int blo, int bhi) {
    Match best{alo, blo, 0};
    std::unordered_map<int, int> j2len;
    for (int i = alo; i < ahi; ++i) {
        std::unordered_map<int, int> new_j2len;
        for (int j = blo; j < bhi; ++j) {
            if (a[i] != b[j]) continue;
            int k = 1;
            auto it = j2len.find(j - 1);
            if (it != j2len.end()) k = it->second + 1;
            new_j2len[j] = k;
            if (k > best.size) {
                best.a = i - k + 1;
                best.b = j - k + 1;
                best.size = k;
            }
        }
        j2len = std::move(new_j2len);
    }
    return best;
}

// Total matched characters between a[alo,ahi) and b[blo,bhi), recursing
// into the unmatched left/right remainders around each longest match.
int count_matches(const std::string& a, int alo, int ahi,
                   const std::string& b, int blo, int bhi) {
    if (alo >= ahi || blo >= bhi) return 0;
    Match m = find_longest_match(a, alo, ahi, b, blo, bhi);
    if (m.size == 0) return 0;
    int total = m.size;
    total += count_matches(a, alo, m.a, b, blo, m.b);
    total += count_matches(a, m.a + m.size, ahi, b, m.b + m.size, bhi);
    return total;
}

// Ratcliff/Obershelp similarity ratio, in [0, 1] -- the same algorithm
// Python's difflib.SequenceMatcher.ratio() uses, minus the "autojunk"
// heuristic, which only applies to sequences of 200+ elements and is
// irrelevant for short UI labels.
double sequence_ratio(const std::string& a, const std::string& b) {
    int total_len = static_cast<int>(a.size() + b.size());
    if (total_len == 0) return 1.0;
    int matches = count_matches(a, 0, static_cast<int>(a.size()), b, 0, static_cast<int>(b.size()));
    return 2.0 * matches / total_len;
}

// Splits on anything that is not a letter or digit, so punctuation, hyphens
// and the newlines between boxes are all just separators.
std::vector<std::string> words_of(const std::string& s) {
    std::vector<std::string> words;
    std::string current;
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            current += static_cast<char>(c);
        } else if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) words.push_back(current);
    return words;
}

std::string join(const std::vector<std::string>& words, std::size_t start, std::size_t count) {
    std::string out;
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) out += ' ';
        out += words[start + i];
    }
    return out;
}

std::vector<std::string> lower_all(const std::vector<std::string>& words) {
    std::vector<std::string> out(words.size());
    std::transform(words.begin(), words.end(), out.begin(), to_lower);
    return out;
}

// Best score for the keyword's words against any run of the same length in the
// document's words, and the run that produced it. Sizing the comparison to the
// keyword's own length is what keeps a short keyword from being diluted by a
// long line of surrounding text. An identical run scores 1.0 without being
// compared character by character.
KeywordMatch best_window(const std::vector<std::string>& kw_words,
                          const std::vector<std::string>& doc_words,
                          const std::vector<std::string>& doc_lower) {
    KeywordMatch best;
    if (kw_words.empty() || kw_words.size() > doc_lower.size()) return best;
    std::string kw_joined = join(kw_words, 0, kw_words.size());
    for (std::size_t start = 0; start + kw_words.size() <= doc_lower.size(); ++start) {
        std::string window = join(doc_lower, start, kw_words.size());
        double score = (window == kw_joined) ? 1.0 : sequence_ratio(window, kw_joined);
        if (score > best.score) {
            best.score = score;
            best.text = join(doc_words, start, kw_words.size());
        }
        if (best.score == 1.0) break;
    }
    return best;
}

}  // namespace

bool overlaps(const DetectedBox& b, const Region& region) {
    // bounds taken across all four corners: a tilted box's corner 0 is not necessarily its left edge
    int bx0 = b.box[0][0], bx1 = b.box[0][0];
    int by0 = b.box[0][1], by1 = b.box[0][1];
    for (const auto& p : b.box) {
        bx0 = std::min(bx0, p[0]);
        bx1 = std::max(bx1, p[0]);
        by0 = std::min(by0, p[1]);
        by1 = std::max(by1, p[1]);
    }
    return bx0 < region.x + region.w && bx1 > region.x &&
           by0 < region.y + region.h && by1 > region.y;
}

std::pair<const DetectedBox*, double> find_by_keyword(const std::string& keyword,
                                                       const std::vector<DetectedBox>& boxes,
                                                       const Region& region) {
    const DetectedBox* best = nullptr;
    double best_score = 0.0;
    std::vector<std::string> kw_words = words_of(to_lower(keyword));
    for (const auto& b : boxes) {
        if (!overlaps(b, region)) continue;
        std::vector<std::string> doc_words = words_of(b.text);
        KeywordMatch m = best_window(kw_words, doc_words, lower_all(doc_words));
        if (m.score > best_score) {
            best_score = m.score;
            best = &b;
        }
    }
    return {best, best_score};
}

KeywordMatch find_keyword_in_region(const std::vector<std::string>& keywords,
                                     const std::vector<DetectedBox>& boxes,
                                     const Region& region) {
    std::string joined;
    for (const auto& b : boxes) {
        if (!overlaps(b, region)) continue;
        if (!joined.empty()) joined += '\n';
        joined += b.text;
    }
    std::vector<std::string> doc_words = words_of(joined);
    std::vector<std::string> doc_lower = lower_all(doc_words);

    KeywordMatch best;
    for (const auto& kw : keywords) {
        KeywordMatch m = best_window(words_of(to_lower(kw)), doc_words, doc_lower);
        if (m.score > best.score) best = m;
        if (best.score == 1.0) break;
    }
    return best;
}

std::pair<int, int> box_center(const std::array<std::array<int, 2>, 4>& box) {
    int sx = 0, sy = 0;
    for (const auto& p : box) {
        sx += p[0];
        sy += p[1];
    }
    return {sx / 4, sy / 4};
}

std::pair<int, int> region_center(const Region& region) {
    return {region.x + region.w / 2, region.y + region.h / 2};
}
