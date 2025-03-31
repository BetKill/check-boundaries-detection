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

/**
 * @brief Calculates the Intersection over Union (IoU) between two polygons.
 *
 * The function creates binary masks for two polygons and computes the area of intersection
 * and the area of union to calculate the IoU metric.
 *
 * @param points1 A vector of points representing the first polygon.
 * @param points2 A vector of points representing the second polygon.
 *
 * @return The IoU metric, calculated as the area of intersection divided by the area of union.
 */
double calculateIoU(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    // Create masks for the polygons
    cv::Mat mask1 = cv::Mat::zeros(1000, 1000, CV_8UC1);  // You can adjust the size
    cv::Mat mask2 = cv::Mat::zeros(1000, 1000, CV_8UC1);

    std::vector<cv::Point> contour1, contour2;
    for (const auto& p : points1) contour1.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    for (const auto& p : points2) contour2.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));

    // Draw polygons on the masks
    cv::fillPoly(mask1, std::vector<std::vector<cv::Point>>{contour1}, cv::Scalar(255));
    cv::fillPoly(mask2, std::vector<std::vector<cv::Point>>{contour2}, cv::Scalar(255));

    // Calculate intersection and union areas
    cv::Mat intersection;
    cv::bitwise_and(mask1, mask2, intersection);
    cv::Mat unionMat;
    cv::bitwise_or(mask1, mask2, unionMat);

    double intersectionArea = cv::countNonZero(intersection);
    double unionArea = cv::countNonZero(unionMat);

    return intersectionArea / unionArea;  // IoU = Area of Intersection / Area of Union
}

/**
 * @brief Loads ground truth data (polygons) from a JSON annotation file.
 *
 * This function reads a JSON file containing annotations and extracts the polygon points
 * for each annotated region in the images.
 *
 * @param fileName Path to the JSON annotation file.
 *
 * @return A map of filenames to their associated ground truth polygons.
 */
std::map<std::string, std::vector<std::vector<cv::Point2f>>> loadGroundTruth(const std::string& fileName) {
    std::map<std::string, std::vector<std::vector<cv::Point2f>>> groundTruth;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Unable to open the annotation file: " << fileName << std::endl;
        return groundTruth;
    }

    json data;
    file >> data;

    // Iterate through the items in the JSON
    for (auto& item : data.items()) {
        // Extract the filename from the JSON value (this will be the value of the "filename" key)
        std::string fileName = item.value()["filename"];

        // Extract the regions
        json regions = item.value()["regions"];

        for (auto& region : regions) {
            std::vector<cv::Point2f> points;
            std::vector<int> pointsX = region["shape_attributes"]["all_points_x"];
            std::vector<int> pointsY = region["shape_attributes"]["all_points_y"];

            for (size_t i = 0; i < pointsX.size(); ++i) {
                points.push_back(cv::Point2f(pointsX[i], pointsY[i]));
            }

            // Add the polygon to the list for this file
            groundTruth[fileName].push_back(points);
        }
    }
    return groundTruth;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // Path to the annotation JSON file
    std::string annotationFile = "../annotation.json";
    std::map<std::string, std::vector<std::vector<cv::Point2f>>> groundTruth = loadGroundTruth(annotationFile);

    // Path to the folder containing the images
    std::string imageFolder = "../images";
    std::vector<std::string> imageFiles;

    if (!std::filesystem::exists(imageFolder)) {
        std::cerr << "Directory not found: " << imageFolder << std::endl;
        return -1;
    }

    // List all image files with .jpg extension
    for (const auto& entry : std::filesystem::directory_iterator(imageFolder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            imageFiles.push_back(entry.path().string());
        }
    }

    std::cout << "Found images: " << imageFiles.size() << std::endl;

    // Path to the output file where results will be saved
    std::string outputFile = "../images/result.txt";
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << outputFile << std::endl;
        return -1;
    }

    // Process each image in the folder
    for (const auto& imagePath : imageFiles) {
        std::string fileName = std::filesystem::path(imagePath).filename().string();

        // Remove extra digits after .jpg
        size_t pos = fileName.find(".jpg");
        if (pos != std::string::npos) {
            fileName = fileName.substr(0, pos + 4);
        }

        std::cout << "Processing image: " << fileName << std::endl;

        // Read the image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Cannot load image: " << imagePath << std::endl;
            outFile << "Cannot load image: " << imagePath << "\n";
            continue;
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // Detect edges using Canny edge detection
        cv::Mat edges;
        cv::Canny(blurred, edges, 50, 150);

        // Find contours in the edge-detected image
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        cv::RotatedRect bestRect;
        bool rectFound = false;

        // Find the largest rectangle in the contours
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

            // Draw the rectangle on the image
            for (int i = 0; i < 4; ++i) {
                cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        } else {
            outFile << "No rectangle found for: " << imagePath << "\n";
        }

        // Check if ground truth exists for this image and draw it
        if (groundTruth.find(fileName) != groundTruth.end()) {
            const std::vector<cv::Point2f>& gtPoints = groundTruth[fileName].front();  // Use the first polygon (if multiple exist)
            std::vector<cv::Point> gtPolygon;
            for (const auto& pt : gtPoints) {
                gtPolygon.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            cv::polylines(image, gtPolygon, true, cv::Scalar(0, 0, 255), 2);  // Draw ground truth in red
        } else {
            std::cout << "No ground truth for: " << fileName << std::endl;
        }

        // Calculate the IoU between the detected points and the ground truth
        double IoU = 0.0;
        if (groundTruth.find(fileName) != groundTruth.end()) {
            IoU = calculateIoU(detectedPoints, groundTruth[fileName].front());  // Compare with the first ground truth polygon
        }

        // Write the result to the output file
        outFile << "Image: " << imagePath << ", IoU: " << IoU << "\n";

        // Optionally display the image
        cv::imshow("Detected", image);
        cv::waitKey(0);
    }

    // Close the output file
    outFile.close();

    return 0;
}
