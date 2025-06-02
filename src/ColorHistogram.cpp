#include "ColorHistogram.hpp"

ColorHistogram::ColorHistogram(int bins) : bins_(bins) {}

cv::Mat ColorHistogram::extract(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int histSize[] = { bins_, bins_, bins_ };
    float hRange[] = { 0, 180 };
    float sRange[] = { 0, 256 };
    float vRange[] = { 0, 256 };
    const float* ranges[] = { hRange, sRange, vRange };
    int channels[] = { 0, 1, 2 };

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);
    cv::normalize(hist, hist);
    hist.convertTo(hist, CV_32F);

    return hist.reshape(1, 1); // Convert to 1D row vector
}
