#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

// CUDA Kernel for Gaussian Blur
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int kernelSize, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weightSum = 0.0f;
    int halfKernel = kernelSize / 2;

    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);

            float distance = kx * kx + ky * ky;
            float weight = expf(-distance / (2 * sigma * sigma));
            sum += input[ny * width + nx] * weight;
            weightSum += weight;
        }
    }
    output[y * width + x] = static_cast<unsigned char>(sum / weightSum);
}

// CUDA Kernel for Unsharp Masking
__global__ void unsharpMaskKernel(unsigned char* original, unsigned char* blurred, unsigned char* output, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int mask = original[idx] - blurred[idx];
    output[idx] = min(max(original[idx] + static_cast<int>(alpha * mask), 0), 255);
}

// Function to calculate Otsu's Threshold (CPU-based for simplicity)
double calculateOtsuThreshold(const Mat& inputImage) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&inputImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    int totalPixels = inputImage.rows * inputImage.cols;
    float sum = 0, sumB = 0, weightB = 0, weightF = 0, maxVariance = 0;
    double threshold = 0;

    for (int i = 0; i < histSize; ++i) sum += i * hist.at<float>(i);

    for (int i = 0; i < histSize; ++i) {
        weightB += hist.at<float>(i);
        if (weightB == 0) continue;

        weightF = totalPixels - weightB;
        if (weightF == 0) break;

        sumB += i * hist.at<float>(i);
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

// CUDA Kernel for Thresholding
__global__ void thresholdKernel(unsigned char* input, unsigned char* output, int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = input[idx] > threshold ? 255 : 0;
}

int main() {
    // Load image in grayscale
    string imagePath = "input.jpeg";
    Mat originalImage = imread(imagePath, IMREAD_GRAYSCALE);
    if (originalImage.empty()) {
        cout << "Error: Could not load the image!" << endl;
        return -1;
    }

    int width = originalImage.cols;
    int height = originalImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate memory for images
    Mat blurredImage(height, width, CV_8UC1);
    Mat sharpenedImage(height, width, CV_8UC1);
    Mat binaryImage(height, width, CV_8UC1);

    unsigned char* d_input, * d_blurred, * d_sharpened, * d_binary;

    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_blurred, imageSize);
    cudaMalloc((void**)&d_sharpened, imageSize);
    cudaMalloc((void**)&d_binary, imageSize);

    cudaMemcpy(d_input, originalImage.data, imageSize, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Step 1: Gaussian Blur
    int kernelSize = 5;
    float sigma = 1.0f;
    gaussianBlurKernel << <gridDim, blockDim >> > (d_input, d_blurred, width, height, kernelSize, sigma);

    // Step 2: Unsharp Masking
    float alpha = 1.5f;
    unsharpMaskKernel << <gridDim, blockDim >> > (d_input, d_blurred, d_sharpened, width, height, alpha);

    // Copy sharpened image back to CPU for Otsu's Threshold calculation
    cudaMemcpy(sharpenedImage.data, d_sharpened, imageSize, cudaMemcpyDeviceToHost);
    double otsuThreshold = calculateOtsuThreshold(sharpenedImage);

    // Step 3: Apply Thresholding
    thresholdKernel << <gridDim, blockDim >> > (d_sharpened, d_binary, width, height, static_cast<unsigned char>(otsuThreshold));

    // Copy results back to CPU
    cudaMemcpy(blurredImage.data, d_blurred, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(binaryImage.data, d_binary, imageSize, cudaMemcpyDeviceToHost);

    // Display and save results
    imshow("Original Image", originalImage);
    imshow("Blurred Image", blurredImage);
    imshow("Sharpened Image", sharpenedImage);
    imshow("Binary Image", binaryImage);

    imwrite("blurred_image.png", blurredImage);
    imwrite("sharpened_image.png", sharpenedImage);
    imwrite("binary_image.png", binaryImage);

    waitKey(0);

    // Free CUDA memory
    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_sharpened);
    cudaFree(d_binary);

    return 0;
}
