#include <stdexcept>

#include "spectromat.h"

Spectromat::Spectromat(unsigned int width, unsigned int height, unsigned int lineHeight)
    : width(width), height(height), lineHeight(lineHeight)
{
    // initialize width x height black image
    for (int i = 0; i < height / lineHeight; i++)
        lines.push_back(cv::Mat::zeros(lineHeight, width, CV_8UC3));
}

cv::Mat Spectromat::stackImages() {
    cv::Mat result(height, width, lines[0].type());  // full image
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::Rect roi(0, height - (i + 1) * lineHeight, width, lineHeight);
        lines[lines.size() - 1 - i].copyTo(result(roi));  // copy lines bottom to top
    }

    return result;
}

cv::Mat Spectromat::pushLineAndPop(cv::Mat newLine) {
    if (newLine.cols != width || newLine.rows != lineHeight)
        throw std::invalid_argument("newLine does not match expected line dimensions");

    lines.push_back(newLine);
    cv::Mat ret = lines[0];
    lines.erase(lines.begin());  // remove from queue
    return ret;
}

std::vector<cv::Mat> Spectromat::getLines() const {
    return lines;
}

unsigned int Spectromat::getWidth() const {
    return width;
}

unsigned int Spectromat::getHeight() const {
    return height;
}

unsigned int Spectromat::getLineHeight() const {
    return lineHeight;
}
