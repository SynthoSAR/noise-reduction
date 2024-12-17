#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to apply Gaussian Blur
Mat applyGaussianBlur(const Mat& inputImage, int kernelSize, double sigma) {
    Mat blurredImage;
    GaussianBlur(inputImage, blurredImage, Size(kernelSize, kernelSize), sigma);
    return blurredImage;
}

// Function to apply Unsharp Masking
Mat applyUnsharpMask(const Mat& inputImage, const Mat& blurredImage, double alpha) {
    Mat mask, sharpenedImage;
    mask = inputImage - blurredImage;
    sharpenedImage = inputImage + alpha * mask;
    return sharpenedImage;
}

// Function to enhance contrast using Histogram Equalization
Mat enhanceContrast(const Mat& inputImage) {
    Mat equalizedImage;
    equalizeHist(inputImage, equalizedImage);
    return equalizedImage;
}

// Function to calculate Otsu's threshold
double calculateOtsuThreshold(const Mat& inputImage) {
    int histSize = 256; // Grayscale image has 256 intensity levels
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&inputImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    int totalPixels = inputImage.rows * inputImage.cols;
    vector<float> histData(histSize);
    for (int i = 0; i < histSize; ++i) {
        histData[i] = hist.at<float>(i);
    }

    float sum = 0, sumB = 0, weightB = 0, weightF = 0, maxVariance = 0;
    double threshold = 0;

    for (int i = 0; i < histSize; ++i) {
        sum += i * histData[i];
    }

    for (int i = 0; i < histSize; ++i) {
        weightB += histData[i];
        if (weightB == 0) continue;
        weightF = totalPixels - weightB;
        if (weightF == 0) break;

        sumB += i * histData[i];
        float meanB = sumB / weightB;
        float meanF = (sum - sumB) / weightF;
        float betweenClassVariance = weightB * weightF * (meanB - meanF) * (meanB - meanF);

        if (betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            threshold = i;
        }
    }
    return threshold;
}

// Function to apply Otsu's binary thresholding
Mat applyOtsuThresholding(const Mat& inputImage, double otsuThreshold) {
    Mat binaryImage;
    threshold(inputImage, binaryImage, otsuThreshold, 255, THRESH_BINARY);
    return binaryImage;
}

// Main function
int main() {
    // Load the SAR image in grayscale
    string imagePath = "input.jpeg"; // Replace with your image path
    Mat originalImage = imread(imagePath, IMREAD_GRAYSCALE);
    if (originalImage.empty()) {
        cout << "Error: Could not load the image!" << endl;
        return -1;
    }

    // Step 1: Apply Gaussian Blur
    int kernelSize = 5; // Kernel size for Gaussian Blur
    double sigma = 1.0; // Standard deviation for Gaussian Blur
    Mat blurredImage = applyGaussianBlur(originalImage, kernelSize, sigma);

    // Step 2: Apply Unsharp Masking
    double alpha = 1.5; // Weight for unsharp masking
    Mat sharpenedImage = applyUnsharpMask(originalImage, blurredImage, alpha);

    // Step 3: Enhance Contrast
    Mat contrastImage = enhanceContrast(sharpenedImage);

    // Step 4: Calculate Otsu's Threshold
    double otsuThreshold = calculateOtsuThreshold(contrastImage);

    // Step 5: Apply Otsu's Binary Thresholding
    Mat binaryImage = applyOtsuThresholding(contrastImage, otsuThreshold);

    // Display results
    imshow("Original Image", originalImage);
    imshow("Blurred Image", blurredImage);
    imshow("Sharpened Image", sharpenedImage);
    imshow("Contrast Enhanced Image", contrastImage);
    imshow("Binary Image", binaryImage);

    // Save results
    imwrite("blurred_image.png", blurredImage);
    imwrite("sharpened_image.png", sharpenedImage);
    imwrite("contrast_image.png", contrastImage);
    imwrite("binary_image.png", binaryImage);

    waitKey(0);
    return 0;
}
