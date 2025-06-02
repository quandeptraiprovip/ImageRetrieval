#pragma once
#include "FeatureExtractor.hpp"
#include <opencv2/opencv.hpp>

class ColorHistogram : public FeatureExtractor{
public:
    ColorHistogram(int bins = 32);  // Số lượng bins cho mỗi kênh
    cv::Mat extract(const cv::Mat& image);   // Trả về vector đặc trưng

private:
    int bins_;
};
