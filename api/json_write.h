#pragma once
#include <string>

// Escapes a string for embedding inside a JSON string literal. Handles what
// this API's own output can actually contain -- OCR text and alarm text --
// (quotes, backslashes, and control characters), not general-purpose JSON.
std::string json_escape(const std::string& s);
