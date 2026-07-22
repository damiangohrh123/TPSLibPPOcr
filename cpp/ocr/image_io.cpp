#include "image_io.h"
#include <cstdio>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include "stb_image.h"

namespace {

bool ends_with(const std::string& path, const std::string& suffix) {
	return path.size() >= suffix.size() &&
	       path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Parses "..._<width>x<height>.bgr888" -> {width, height}, or {0, 0} if the
// file name doesn't match that shape.
std::pair<int, int> parse_raw_dimensions(const std::string& path) {
	size_t ext = path.rfind(".bgr888");
	if (ext == std::string::npos) return {0, 0};
	size_t underscore = path.rfind('_', ext);
	if (underscore == std::string::npos) return {0, 0};
	std::string dims = path.substr(underscore + 1, ext - underscore - 1);
	size_t x = dims.find('x');
	if (x == std::string::npos) return {0, 0};
	int width = std::atoi(dims.substr(0, x).c_str());
	int height = std::atoi(dims.substr(x + 1).c_str());
	return {width, height};
}

}  // namespace

// Loads a raw BGR888 file straight into a cv::Mat -- no decoding, just a
// direct read, since the bytes on disk are already exactly what a CV_8UC3
// Mat holds in memory.
cv::Mat load_raw_bgr888(const std::string& path) {
	auto [width, height] = parse_raw_dimensions(path);
	if (width <= 0 || height <= 0) {
		fprintf(stderr, "failed to parse dimensions from %s (expected ..._<width>x<height>.bgr888)\n",
			path.c_str());
		return cv::Mat();
	}
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		fprintf(stderr, "failed to open %s\n", path.c_str());
		return cv::Mat();
	}
	cv::Mat img(height, width, CV_8UC3);
	size_t expected_bytes = img.total() * img.elemSize();
	f.read(reinterpret_cast<char*>(img.data), expected_bytes);
	if (!f) {
		fprintf(stderr, "failed to read %s: expected %zu bytes for %dx%d\n",
			path.c_str(), expected_bytes, width, height);
		return cv::Mat();
	}
	return img;
}

// Decodes an image file to BGR; returns an empty Mat on failure.
cv::Mat load_image(const std::string& path) {
	if (ends_with(path, ".bgr888")) {
		return load_raw_bgr888(path);
	}
	int width = 0, height = 0, channels_in_file = 0;
	unsigned char* rgb = stbi_load(path.c_str(), &width, &height, &channels_in_file, /*desired_channels=*/3);
	if (!rgb) {
		fprintf(stderr, "failed to decode %s: %s\n", path.c_str(), stbi_failure_reason());
		return cv::Mat();
	}
	cv::Mat rgb_mat(height, width, CV_8UC3, rgb);
	cv::Mat bgr_mat;
	cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
	stbi_image_free(rgb);
	return bgr_mat;
}

// Decodes an in-memory image buffer to BGR; returns an empty Mat on failure.
cv::Mat decode_image_from_memory(const unsigned char* data, size_t len) {
	int width = 0, height = 0, channels_in_file = 0;
	unsigned char* rgb = stbi_load_from_memory(data, static_cast<int>(len), &width, &height,
		&channels_in_file, /*desired_channels=*/3);
	if (!rgb) {
		fprintf(stderr, "failed to decode in-memory image: %s\n", stbi_failure_reason());
		return cv::Mat();
	}
	cv::Mat rgb_mat(height, width, CV_8UC3, rgb);
	cv::Mat bgr_mat;
	cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
	stbi_image_free(rgb);
	return bgr_mat;
}
