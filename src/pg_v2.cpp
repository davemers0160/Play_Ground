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
#include "ssim.h"
#include "dlib_matrix_threshold.h"

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
        

        dlib::matrix<uint16_t> gt_img, tmp, tmp2, dm_img;
        dlib::matrix<float> ssim_map;

        // load in the images --  not done in real life but for here
        std::string gt_image_name = "D:/IUPUI/Test_Data/rw/WS2/lidar/lidar_rng_right_00000_8bit.png";
        std::string dm_image_name = "D:/IUPUI/PhD/Results/dfd_dnn_pso/itr1/dfd_pso_13/depthmap_image_v6_pso_13_01_test_00049.png";

        dlib::load_image(dm_img, dm_image_name);
        dlib::load_image(tmp, gt_image_name);

        tmp2.set_size(dm_img.nr(), dm_img.nc());
        dlib::set_subm(tmp2, 0, 0, dm_img.nr(), dm_img.nc()) = dlib::subm(tmp, 0, 0, dm_img.nr(), dm_img.nc());
                
        truncate_threshold(tmp2, gt_img, 255);

        auto s1 = dlib::sum(dlib::matrix_cast<float>(gt_img));
        auto s2 = dlib::sum(dlib::matrix_cast<float>(dm_img));

        auto m_t = dlib::mean(dlib::matrix_cast<float>(gt_img));
        uint16_t m = (uint16_t)(std::floor(m_t));
        dlib::matrix<uint16_t> test_img = dlib::uniform_matrix<uint16_t>(dm_img.nr(), dm_img.nc(), m);

        dlib::matrix<uint16_t> t2 = dlib::uniform_matrix<uint16_t>(256, 256, 1);

        auto s1_t2 = dlib::sum(t2);
        auto s2_t2 = dlib::sum(dlib::matrix_cast<float>(t2));

        auto m1_t2 = dlib::mean(t2);
        auto m2_t2 = dlib::mean(dlib::matrix_cast<float>(t2));

        double ssim_dm = ssim(gt_img, dm_img, ssim_map);
        double ssim_test = ssim(gt_img, test_img, ssim_map);

        dlib::matrix<float> sub_dm = dlib::matrix_cast<float>(gt_img) - dlib::matrix_cast<float>(dm_img);
        dlib::matrix<float> sub_test = dlib::matrix_cast<float>(gt_img) - dlib::matrix_cast<float>(test_img);

        double m1_dm = dlib::mean(dlib::abs(sub_dm));
        double m2_dm = std::sqrt(dlib::mean(dlib::squared(sub_dm)));

        double m1_test = dlib::mean(dlib::abs(sub_test));
        double m2_test = std::sqrt(dlib::mean(dlib::squared(sub_test)));

        //double var = dlib::variance(gt[idx]);
        double rng = 255;
        double nmae_dm = m1_dm / rng;
        double nrmse_dm = m2_dm / rng;
        double nmae_te = m1_test / rng;
        double nrmse_te = m2_test / rng;

        //rmse_val += std::sqrt(m2) / rng;

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

