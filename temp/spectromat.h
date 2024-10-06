#include <opencv2/opencv.hpp>
#include <vector>

class Spectromat {
public:
    Spectromat(unsigned int width, unsigned int height, unsigned int lineHeight);

    cv::Mat stackImages();
    cv::Mat pushLineAndPop(cv::Mat newLine);

    std::vector<cv::Mat> getLines() const;
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getLineHeight() const;

private:
    std::vector<cv::Mat> lines;
    unsigned int width;
    unsigned int height;
    unsigned int lineHeight;
};
