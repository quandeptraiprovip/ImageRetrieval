#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class KCluster {
public:
    void build(const cv::Mat& allFeatures, int clusterCount);
    std::vector<int> getCandidateIndices(const cv::Mat& queryFeature, int topKClusters = 1);
    void save(const std::string& path);
    void load(const std::string& path);
    void savePerClusterBinary(const std::string& folder, const std::vector<std::string>& imagePaths);
    static void loadClusterBinary(const std::string& clusterFile, std::vector<std::string>& imageNames, std::vector<cv::Mat>& features);
    const std::vector<std::vector<int>>& getClusters() const {
        return clusters_;
    }

private:
    cv::Mat centers_;
    std::vector<std::vector<int>> clusters_;
    cv::Mat allFeatures_;
};