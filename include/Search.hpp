#pragma once
#include <opencv2/opencv.hpp>
#include "FeatureExtractor.hpp"
#include "KCluster.hpp"
#include <string>
#include <vector>

class Search {
public:
    Search(const std::string& centroidPath, const std::string& clusterFolder);

    std::string query(const cv::Mat& image, FeatureExtractor* extractor);

private:
    cv::Mat centroids_;
    std::string clusterFolder_;

    int findClosestCluster(const cv::Mat& feature);
    std::string findBestMatchInCluster(const std::string& clusterFile, const cv::Mat& queryFeature);
};
