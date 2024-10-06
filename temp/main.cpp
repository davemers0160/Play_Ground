#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <random>
#include "spectromat.h"

cv::Mat generateRandomLine(int width, int height) {
    cv::Mat line(height, width, CV_8UC3);
    cv::RNG rng(cv::getTickCount());

    for (int i = 0; i < width; ++i) {
        uchar value = rng.uniform(0, 2) * 255;
        for (int l = 0; l < height; l++)
            line.at<cv::Vec3b>(l, i) = cv::Vec3b(value, value, value);
    }

    return line;
}

int main() {
    int width = 640;
    int height = 200;
    int lineHeight = 2;
    Spectromat spectro(width, height, lineHeight);
    int wait_ms = 100;

    cv::String windowTitle = "Spectrogram";
    while (true) {
        // generate a new line and push
        cv::Mat newLine = generateRandomLine(width, lineHeight);
        spectro.pushLineAndPop(newLine);

        // get image
        cv::Mat stackedImage = spectro.stackImages();
        cv::imshow(windowTitle, stackedImage);

        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms / 2));

        // break if window closed or key pressed
        if (cv::waitKey(wait_ms / 2) >= 0
            || cv::getWindowProperty(windowTitle, cv::WND_PROP_VISIBLE) < 1)
            break;
    }

    return 0;
}
