#include "KCluster.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>

void KCluster::build(const cv::Mat& allFeatures, int clusterCount) {
    allFeatures_ = allFeatures;

    // Thực hiện KMeans
    cv::Mat labels;
    cv::kmeans(allFeatures, clusterCount, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers_);

    // Khởi tạo các cụm
    clusters_.resize(clusterCount);
    for (int i = 0; i < labels.rows; ++i) {
        int clusterIdx = labels.at<int>(i, 0);
        clusters_[clusterIdx].push_back(i);  // Lưu index của ảnh thuộc cụm
    }
}

std::vector<int> KCluster::getCandidateIndices(const cv::Mat& queryFeature, int topKClusters) {
    // Tính khoảng cách từ queryFeature đến các center
    std::vector<std::pair<double, int>> distances;

    for (int i = 0; i < centers_.rows; ++i) {
        double dist = cv::norm(queryFeature, centers_.row(i), cv::NORM_L2);
        distances.emplace_back(dist, i);
    }

    // Sắp xếp và chọn top K cluster gần nhất
    std::sort(distances.begin(), distances.end());
    std::vector<int> candidateIndices;
    for (int i = 0; i < topKClusters && i < distances.size(); ++i) {
        int clusterIdx = distances[i].second;
        candidateIndices.insert(candidateIndices.end(),
                                clusters_[clusterIdx].begin(), clusters_[clusterIdx].end());
    }
    return candidateIndices;
}

void KCluster::save(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    // Ghi số dòng, số cột, kiểu dữ liệu
    int rows = centers_.rows;
    int cols = centers_.cols;
    int type = centers_.type();
    ofs.write((char*)&rows, sizeof(int));
    ofs.write((char*)&cols, sizeof(int));
    ofs.write((char*)&type, sizeof(int));

    // Ghi dữ liệu thực tế
    size_t dataSize = centers_.total() * centers_.elemSize();
    ofs.write((char*)centers_.data, dataSize);



    ofs.close();
}


void KCluster::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }

    int rows, cols, type;
    ifs.read((char*)&rows, sizeof(int));
    ifs.read((char*)&cols, sizeof(int));
    ifs.read((char*)&type, sizeof(int));

    centers_.create(rows, cols, type);
    size_t dataSize = centers_.total() * centers_.elemSize();
    ifs.read((char*)centers_.data, dataSize);

    ifs.close();
}

void KCluster::savePerClusterBinary(const std::string& folder, const std::vector<std::string>& imagePaths) {
    namespace fs = std::filesystem;
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    for (size_t i = 0; i < clusters_.size(); ++i) {
        std::string filePath = folder + "/cluster_" + std::to_string(i) + ".bin";
        std::ofstream out(filePath, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Không thể tạo file: " << filePath << std::endl;
            continue;
        }

        // Ghi số lượng ảnh trong cụm
        int numImages = static_cast<int>(clusters_[i].size());
        out.write(reinterpret_cast<char*>(&numImages), sizeof(int));

        for (int idx : clusters_[i]) {
            const std::string& name = imagePaths[idx];
            int nameLength = static_cast<int>(name.size());

            // Ghi độ dài và tên ảnh
            out.write(reinterpret_cast<char*>(&nameLength), sizeof(int));
            out.write(name.c_str(), nameLength);

            // Ghi đặc trưng tương ứng
            const cv::Mat& feature = allFeatures_.row(idx); // 1 dòng
            int dims = feature.cols;
            out.write(reinterpret_cast<char*>(&dims), sizeof(int));
            out.write(reinterpret_cast<const char*>(feature.ptr<float>()), sizeof(float) * dims);
        }

        out.close();
    }
}


void KCluster::loadClusterBinary(const std::string& clusterFile,
                             std::vector<std::string>& imageNames,
                             std::vector<cv::Mat>& features) {
    std::ifstream in(clusterFile, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Không thể mở file: " << clusterFile << std::endl;
        return;
    }

    int numImages;
    in.read(reinterpret_cast<char*>(&numImages), sizeof(int));

    for (int i = 0; i < numImages; ++i) {
        int nameLen;
        in.read(reinterpret_cast<char*>(&nameLen), sizeof(int));
        if (nameLen <= 0 || nameLen > 1024) {
            std::cerr << "Tên ảnh không hợp lệ. nameLen=" << nameLen << std::endl;
            break;
        }

        std::string name(nameLen, '\0');
        in.read(&name[0], nameLen);
        imageNames.push_back(name);

        int dims;
        in.read(reinterpret_cast<char*>(&dims), sizeof(int));
        // if (dims <= 0 || dims > 10000) {
        //     std::cerr << "Số chiều đặc trưng không hợp lệ: " << dims << std::endl;
        //     break;
        // }

        std::vector<float> buffer(dims);
        in.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * dims);
        cv::Mat feature(1, dims, CV_32F, buffer.data());
        features.push_back(feature.clone()); // clone để giữ bộ nhớ riêng
    }

    in.close();
}
