#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class FeatureExtractor {
public:
    virtual cv::Mat extract(const cv::Mat& image) = 0;
    virtual ~FeatureExtractor() = default;
};