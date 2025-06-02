#pragma once
#include "FeatureExtractor.hpp"
#include "KCluster.hpp"

class ImageDatabase {
public:
    void loadImages(const std::string& folderPath);
    void buildIndex(FeatureExtractor* extractor, int clusterCount);
    std::vector<std::string> query(const cv::Mat& image, FeatureExtractor* extractor, int topK = 5);
    const std::vector<std::string>& getImagePaths() const {
        return imagePaths_;
    }

private:
    std::vector<std::string> imagePaths_;
    KCluster indexer_;
};

