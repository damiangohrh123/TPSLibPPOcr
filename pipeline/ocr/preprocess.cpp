#include "preprocess.h"

NormalizeImage::NormalizeImage(double scale) : scale_(scale) {}

cv::Mat NormalizeImage::operator()(const cv::Mat& img) const {
	// convertTo switches to float and multiplies by scale_ in one pass
	cv::Mat out;
	img.convertTo(out, CV_32FC3, scale_);
	return out;
}
