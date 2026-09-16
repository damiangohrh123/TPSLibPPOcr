#include "preprocess.h"

NormalizeImage::NormalizeImage(double scale) : scale_(scale) {}

cv::Mat NormalizeImage::operator()(const cv::Mat& img) const {
	// convertTo switches to float and multiplies by scale_ in one pass
	cv::Mat out;
	img.convertTo(out, CV_32FC3, scale_);
	return out;
}

std::vector<float> to_nhwc_batch(const cv::Mat& img) {
	CV_Assert(img.type() == CV_32FC3);  // must already be normalized
	cv::Mat contiguous = img.isContinuous() ? img : img.clone();  // need one unbroken buffer to read raw floats from
	const float* data = contiguous.ptr<float>(0);
	size_t n = static_cast<size_t>(contiguous.total()) * static_cast<size_t>(contiguous.channels());
	return std::vector<float>(data, data + n);
}
