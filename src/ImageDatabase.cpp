#include "ImageDatabase.hpp"
#include "KCluster.hpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace fs = std::filesystem;

void ImageDatabase::loadImages(const std::string& folderPath) {
    imagePaths_.clear();
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            imagePaths_.push_back(entry.path().string());
        }
    }
}

void ImageDatabase::buildIndex(FeatureExtractor* extractor, int clusterCount) {
    cv::Mat allFeatures;

    for (const auto& path : imagePaths_) {
        cv::Mat img = cv::imread(path);
        if (!img.empty()) {
            cv::Mat feature = extractor->extract(img);
            allFeatures.push_back(feature);
        }
    }

    indexer_.build(allFeatures, clusterCount);
}

// std::vector<std::string> ImageDatabase::query(const cv::Mat& image, FeatureExtractor* extractor, int topK) {
//     cv::Mat queryFeature = extractor->extract(image);
//     std::vector<int> candidateIndices = indexer_.getCandidateIndices(queryFeature, topK * 5);  // lấy rộng hơn để chọn topK thật

//     std::vector<std::pair<double, int>> distances;
//     for (int idx : candidateIndices) {
//         cv::Mat dbFeature = indexer_.getFeature(idx);
//         double dist = cv::norm(queryFeature, dbFeature, cv::NORM_L2);
//         distances.emplace_back(dist, idx);
//     }

//     std::sort(distances.begin(), distances.end());

//     std::vector<std::string> results;
//     for (int i = 0; i < std::min(topK, (int)distances.size()); ++i) {
//         results.push_back(imagePaths_[distances[i].second]);
//     }

//     return results;
// }
