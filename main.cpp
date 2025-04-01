#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <windows.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

double calculateIoU(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    cv::Mat mask1 = cv::Mat::zeros(1000, 1000, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(1000, 1000, CV_8UC1);

    std::vector<cv::Point> contour1, contour2;
    for (const auto& p : points1) contour1.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    for (const auto& p : points2) contour2.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));

    cv::fillPoly(mask1, std::vector<std::vector<cv::Point>>{contour1}, cv::Scalar(255));
    cv::fillPoly(mask2, std::vector<std::vector<cv::Point>>{contour2}, cv::Scalar(255));

    cv::Mat intersection, unionMat;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, unionMat);

    double intersectionArea = cv::countNonZero(intersection);
    double unionArea = cv::countNonZero(unionMat);

    return intersectionArea / unionArea;
}

std::map<std::string, std::vector<std::vector<cv::Point2f>>> loadGroundTruth(const std::string& fileName) {
    std::map<std::string, std::vector<std::vector<cv::Point2f>>> groundTruth;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Unable to open annotation file: " << fileName << std::endl;
        return groundTruth;
    }

    json data;
    file >> data;

    for (auto& item : data.items()) {
        std::string fileName = item.value()["filename"];
        json regions = item.value()["regions"];

        for (auto& region : regions) {
            std::vector<cv::Point2f> points;
            std::vector<int> pointsX = region["shape_attributes"]["all_points_x"];
            std::vector<int> pointsY = region["shape_attributes"]["all_points_y"];

            for (size_t i = 0; i < pointsX.size(); ++i) {
                points.push_back(cv::Point2f(pointsX[i], pointsY[i]));
            }

            groundTruth[fileName].push_back(points);
        }
    }
    return groundTruth;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <annotation.json> <image_folder> <output_file>\n";
        return -1;
    }

    std::string annotationFile = argv[1];
    std::string imageFolder = argv[2];
    std::string outputFile = argv[3];

    std::map<std::string, std::vector<std::vector<cv::Point2f>>> groundTruth = loadGroundTruth(annotationFile);
    std::vector<std::string> imageFiles;

    if (!std::filesystem::exists(imageFolder)) {
        std::cerr << "Directory not found: " << imageFolder << std::endl;
        return -1;
    }

    for (const auto& entry : std::filesystem::directory_iterator(imageFolder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            imageFiles.push_back(entry.path().string());
        }
    }

    std::cout << "Found images: " << imageFiles.size() << std::endl;

    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << outputFile << std::endl;
        return -1;
    }

    for (const auto& imagePath : imageFiles) {
        std::string fileName = std::filesystem::path(imagePath).filename().string();
        size_t pos = fileName.find(".jpg");
        if (pos != std::string::npos) {
            fileName = fileName.substr(0, pos + 4);
        }

        std::cout << "Processing image: " << fileName << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Cannot load image: " << imagePath << std::endl;
            outFile << "Cannot load image: " << imagePath << "\n";
            continue;
        }

        cv::Mat gray, blurred, edges;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::Canny(blurred, edges, 50, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        cv::RotatedRect bestRect;
        bool rectFound = false;

        for (const auto& contour : contours) {
            cv::RotatedRect rect = cv::minAreaRect(contour);
            double area = rect.size.width * rect.size.height;
            if (area > maxArea) {
                maxArea = area;
                bestRect = rect;
                rectFound = true;
            }
        }

        std::vector<cv::Point2f> detectedPoints;
        if (rectFound) {
            cv::Point2f vertices[4];
            bestRect.points(vertices);
            for (int i = 0; i < 4; ++i) {
                detectedPoints.push_back(vertices[i]);
            }
            for (int i = 0; i < 4; ++i) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        } else {
            outFile << "No rectangle found for: " << imagePath << "\n";
        }

        if (groundTruth.find(fileName) != groundTruth.end()) {
            const std::vector<cv::Point2f>& gtPoints = groundTruth[fileName].front();
            std::vector<cv::Point> gtPolygon;
            for (const auto& pt : gtPoints) {
                gtPolygon.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            cv::polylines(image, gtPolygon, true, cv::Scalar(0, 0, 255), 2);
        } else {
            std::cout << "No ground truth for: " << fileName << std::endl;
        }

        double IoU = 0.0;
        if (groundTruth.find(fileName) != groundTruth.end()) {
            IoU = calculateIoU(detectedPoints, groundTruth[fileName].front());
        }

        outFile << "Image: " << imagePath << ", IoU: " << IoU << "\n";
        cv::imshow("Detected", image);
        cv::waitKey(0);
    }

    outFile.close();
    return 0;
}
