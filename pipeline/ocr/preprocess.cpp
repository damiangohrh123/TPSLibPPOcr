#include "preprocess.h"

NormalizeImage::NormalizeImage(double scale, const std::array<double, 3>& mean, const std::array<double, 3>& std)
	: scale_(scale), mean_(mean[0], mean[1], mean[2]), std_(std[0], std[1], std[2]) {}

cv::Mat NormalizeImage::operator()(const cv::Mat& img) const {
	cv::Mat scaled;
	img.convertTo(scaled, CV_32FC3, scale_);

	cv::Mat out;
	cv::subtract(scaled, mean_, out);
	cv::divide(out, std_, out);
	return out;
}

std::vector<float> to_nhwc_batch(const cv::Mat& img) {
	CV_Assert(img.type() == CV_32FC3);  // must already be normalized
	cv::Mat contiguous = img.isContinuous() ? img : img.clone();  // need one unbroken buffer to read raw floats from
	const float* data = contiguous.ptr<float>(0);
	size_t n = static_cast<size_t>(contiguous.total()) * static_cast<size_t>(contiguous.channels());
	return std::vector<float>(data, data + n);
}
