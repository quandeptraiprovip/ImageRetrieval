// ClusterSearcher.cpp
#include "Search.hpp"
#include <fstream>
#include <limits>

Search::Search(const std::string& centroidPath, const std::string& clusterFolder)
    : clusterFolder_(clusterFolder) {

    std::ifstream in(centroidPath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Không thể mở file centroid: " << centroidPath << std::endl;
        return;
    }

    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));

    centroids_ = cv::Mat(rows, cols, CV_32F);
    in.read(reinterpret_cast<char*>(centroids_.ptr<float>()), sizeof(float) * rows * cols);

    in.close();
}

int Search::findClosestCluster(const cv::Mat& feature) {
    int bestIdx = 0;
    double minDist = std::numeric_limits<double>::max();
    for (int i = 0; i < centroids_.rows; ++i) {
        double dist = cv::norm(feature, centroids_.row(i));
        if (dist < minDist) {
            minDist = dist;
            bestIdx = i;
        }
    }
    return bestIdx;
}

std::string Search::findBestMatchInCluster(const std::string& clusterFile, const cv::Mat& queryFeature) {
    std::vector<std::string> imageNames;
    std::vector<cv::Mat> features;

    KCluster::loadClusterBinary(clusterFile, imageNames, features);

    float minDist = FLT_MAX;
    std::string bestImage;

    for (size_t i = 0; i < features.size(); ++i) {
        float dist = cv::norm(features[i], queryFeature);
        if (dist < minDist) {
            minDist = dist;
            bestImage = imageNames[i];
        }
    }

    return bestImage;
}


std::string Search::query(const cv::Mat& image, FeatureExtractor* extractor) {
    cv::Mat feature = extractor->extract(image);
    feature.convertTo(feature, CV_32F);
    int clusterId = findClosestCluster(feature);
    std::string clusterFile = clusterFolder_ + "/cluster_" + std::to_string(clusterId) + ".bin";
    return findBestMatchInCluster(clusterFile, feature);
}
