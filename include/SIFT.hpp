#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "FeatureExtractor.hpp"

class SIFT : public FeatureExtractor{
public:
    SIFT(int nFeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, 
                        double edgeThreshold = 10, double sigma = 1.6) {
        sift_ = cv::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    // Hàm trích xuất đặc trưng: trả về descriptor (M x 128 float)
    cv::Mat extract(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3)
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = image;

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        sift_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

        return descriptors; // có thể rỗng nếu không tìm thấy keypoint
    }

private:
    cv::Ptr<cv::SIFT> sift_;
};
