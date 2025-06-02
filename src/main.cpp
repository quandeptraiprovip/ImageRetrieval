#include "SIFT.hpp"
#include "ColorHistogram.hpp"
#include "ImageDatabase.hpp"
#include "KCluster.hpp"
#include "Search.hpp"
#include <iostream>

int main() {
    std::string folderPath = "/Users/minh10hd/Downloads/DataSet/training_images";
    int clusterCount = 10;

    // Load ảnh
    // std::vector<cv::Mat> images;
    ImageDatabase loader;
    loader.loadImages(folderPath);
    // loader.load(images, imagePaths);
    const auto& imagePaths = loader.getImagePaths();

    // Trích xuất đặc trưng
    // ColorHistogram extractor(32);
    SIFT extractor;
    cv::Mat allFeatures;
    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path);
        cv::Mat feature = extractor.extract(img);
        allFeatures.push_back(feature);
    }
    allFeatures.convertTo(allFeatures, CV_32F);


    // Phân cụm
    KCluster indexer;
    indexer.build(allFeatures, clusterCount);

    const auto& clusters = indexer.getClusters();  // bạn cần thêm getter này
    for (int i = 0; i < clusters.size(); ++i) {
        std::cout << "Cluster " << i << ":\n";
        for (int idx : clusters[i]) {
            std::cout << "  - " << imagePaths[idx] << "\n";
        }
    }

    // Lưu thông tin cụm
    indexer.save("centroids.bin");
    indexer.savePerClusterBinary("clusters_bin", imagePaths);



    // std::string queryPath = "/Users/minh10hd/Downloads/DataSet/TestImages/01.jpg";
    // std::string centroidPath = "centroids.bin";
    // std::string clusterFolder = "clusters_bin";

    // ColorHistogram extractor(32);
    // Search searcher(centroidPath, clusterFolder);

    // cv::Mat queryImage = cv::imread(queryPath);
    // if (queryImage.empty()) {
    //     std::cerr << "Could not read image: " << queryPath << std::endl;
    //     return -1;
    // }

    // std::string result = searcher.query(queryImage, &extractor);
    // std::cout << "Best match: " << result << std::endl;



    return 0;
}
