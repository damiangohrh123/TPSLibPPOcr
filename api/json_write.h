#pragma once
#include <string>

// Escapes a string for embedding inside a JSON string literal: quotes,
// backslashes, and control characters. Not general-purpose JSON -- no UTF-8
// validation, and non-ASCII bytes pass through unescaped. Shared by
// ocr_server and step_matcher.
std::string json_escape(const std::string& s);
