#define _CRT_SECURE_NO_WARNINGS

#include "win_network_fcns.h"

//#include <winsock2.h>
//#include <iphlpapi.h>
//
//#pragma comment(lib, "IPHLPAPI.lib")    // Link with Iphlpapi.lib

// C/C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

// dlib includes
#include "dlib/rand.h"
#include "dlib/matrix.h"
#include "dlib/pixel.h"
#include "dlib/image_io.h"
#include "dlib/external/libpng/png.h"
#include "dlib/external/libjpeg/jpeglib.h"
#include "dlib/external/zlib/zlib.h"
#include "dlib/image_transforms.h"
#include <dlib/opencv.h>

// #include "dlib/xml_parser.h"
// #include "dlib/string.h"

// OpenCV includes
#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/video/video.hpp>

// custom includes
//#include "mmaplib.h"
#include "pg.h"
#include "mmap.h"
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "read_binary_image.h" 
#include "make_dir.h"

//#include "pso.h"
//#include "ycrcb_pixel.h"
//#include "dfd_array_cropper.h"
#include "rot_90.h"
#include "dlib_srelu.h"
#include "dlib_elu.h"
#include "center_cropper.h"
#include "dfd_cropper_rw.h"

// Network includes
#include "dfd_net_rw_v6.h"
#include "load_dfd_rw_data.h"

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
extern const uint32_t secondary;
std::string platform;
std::vector<std::array<dlib::matrix<uint16_t>, img_depth>> trn, te, trn_crop, te_crop;
std::vector<dlib::matrix<uint16_t>> gt_train, gt_test, gt_crop, gt_te_crop;

std::string version;
std::string net_name = "dfd_net_";
std::string net_sync_name = "dfd_sync_";
std::string logfileName = "dfd_net_";
std::string gorgon_savefile = "gorgon_dfd_";

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx, jdx;

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    unsigned long training_duration = 1;  // number of hours to train 
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::vector<double> stop_criteria;
    uint64_t num_crops;
    std::pair<uint64_t, uint64_t> crop_size;
    std::vector<std::pair<uint64_t, uint64_t>> crop_sizes = { {1,1}, {38,148} };
    std::vector<uint32_t> filter_num;
    uint64_t max_one_step_count;

    std::vector<std::vector<std::string>> training_file;
    std::string data_directory;
    std::string train_inputfile, test_inputfile;
    std::vector<std::pair<std::string, std::string>> tr_image_files;

    dlib::matrix<uint16_t> g_crop;
    std::array<dlib::matrix<uint16_t>, img_depth> tr_crop;

    std::string platform;
    getPlatform(platform);
    std::cout << "Platform: " << platform << std::endl;

    if (platform.compare(0, 6, "Laptop") == 0)
    {
        std::cout << "Match!" << std::endl;
    }

    try
    {
        int bp = 0;

// ----------------------------------------------------------------------------------------
        
        // load in the images --  not done in real life but for here
        cv::Mat focus_image = cv::imread("D:/IUPUI/Test_Data/rw/Library2/left/exp_40/image_133_40.00.png", CV_LOAD_IMAGE_COLOR);
        cv::Mat defocus_image = cv::imread("D:/IUPUI/Test_Data/rw/Library2/left/exp_40/image_130_40.00.png", CV_LOAD_IMAGE_COLOR);

        // take the image and split the channels
        // then copy each channel into one of the six inputs
        // make sure that the right color gets placed in the right channel
        std::vector<cv::Mat> f(3);
        std::vector<cv::Mat> d(3);
        std::array<dlib::matrix<uint16_t>, img_depth> input_img;

        //for (idx = 0; idx < 3; idx++)
        //{
        //    f[idx] = cv::Mat(imageSize, CV_8UC1);
        //    YCRCB_OUT[idx] = cv::Mat(imageSize, CV_8UC1);
        //}

        //////// Split  images into 3 channels   /////////////////////////////////////////////////////////

        start_time = chrono::system_clock::now();
        cv::split(focus_image, f);
        cv::split(defocus_image, d);

        dlib::cv_image<uint8_t> tmp_img1(f[2]);
        dlib::assign_image(input_img[0], tmp_img1);
        
        dlib::cv_image<uint8_t> tmp_img2(f[1]);
        dlib::assign_image(input_img[1], tmp_img2);

        dlib::cv_image<uint8_t> tmp_img3(f[0]);
        dlib::assign_image(input_img[2], tmp_img3);

        dlib::cv_image<uint8_t> tmp_img4(d[2]);
        dlib::assign_image(input_img[3], tmp_img4);

        dlib::cv_image<uint8_t> tmp_img5(d[1]);
        dlib::assign_image(input_img[4], tmp_img5);

        dlib::cv_image<uint8_t> tmp_img6(d[0]);
        dlib::assign_image(input_img[5], tmp_img6);
        
        stop_time = chrono::system_clock::now();

        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        std::cout << "Test 1: " << elapsed_time.count() << " seconds" << std::endl;

        bp = 1;
        std::array<dlib::matrix<uint16_t>, img_depth> input_img2;
        for (int m = 0; m < 6; ++m)
        {
            input_img2[m].set_size(focus_image.rows, focus_image.cols);
        }

        start_time = chrono::system_clock::now();

        for(idx = 0; idx<focus_image.rows; ++idx)
        {
            for (jdx = 0; jdx < focus_image.cols; ++jdx)
            {
                cv::Vec3w f1 = focus_image.at<cv::Vec3b>(idx, jdx);
                dlib::assign_pixel(input_img2[0](idx, jdx), f1[2]);
                dlib::assign_pixel(input_img2[1](idx, jdx), f1[1]);
                dlib::assign_pixel(input_img2[2](idx, jdx), f1[0]);
                cv::Vec3w d1 = focus_image.at<cv::Vec3b>(idx, jdx);
                dlib::assign_pixel(input_img2[3](idx, jdx), d1[2]);
                dlib::assign_pixel(input_img2[4](idx, jdx), d1[1]);
                dlib::assign_pixel(input_img2[5](idx, jdx), d1[0]);
            }
        }

        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        std::cout << "Test 2: " << elapsed_time.count() << " seconds" << std::endl;

        bp = 2;





    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Press enter to close the program." << std::endl;
        std::cin.ignore();
    }

    std::cin.ignore();
	return 0;

}	// end of main

