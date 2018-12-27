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
//#include "dlib/external/libpng/png.h"
//#include "dlib/external/libjpeg/jpeglib.h"
//#include "dlib/external/zlib/zlib.h"
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
//#include "dlib_srelu.h"
//#include "dlib_elu.h"
#include "center_cropper.h"
#include "dfd_cropper_rw.h"

// Network includes
#include "dfd_net_v14_pso_01.h"
//#include "dfd_net_rw_v18.h"
#include "load_dfd_rw_data.h"

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
extern const uint32_t secondary;
std::string platform;
std::vector<std::array<dlib::matrix<uint16_t>, img_depth>> trn, te, trn_crop, te_crop;
//std::vector<dlib::matrix<uint16_t>> gt_train, gt_test, gt_crop, gt_te_crop;

//std::string version;
std::string net_name = "dfd_net_";
std::string net_sync_name = "dfd_sync_";
std::string logfileName = "dfd_net_";
std::string gorgon_savefile = "gorgon_dfd_";

// ----------------------------------------------------------------------------------------

template <typename img_type1>
void check_matrix(img_type1 img)
{
    if (dlib::is_matrix<img_type1>::value == true)
        std::cout << "matrix" << std::endl;
}

// ----------------------------------------------------------------------------------------

template<typename pixel_type>
void merge_channels(std::array<dlib::matrix<pixel_type>, img_depth> a_img, uint64_t index, dlib::matrix<dlib::rgb_pixel> &img)
{
    uint64_t r, c;

    DLIB_CASSERT(img_depth >= dlib::pixel_traits<dlib::rgb_pixel>::num, "Array depth < " 
        << dlib::pixel_traits<dlib::rgb_pixel>::num);

    uint64_t rows = a_img[0].nr();
    uint64_t cols = a_img[0].nc();

    img.set_size(rows, cols);

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            dlib::rgb_pixel p((uint8_t)(a_img[index+0](r, c)), (uint8_t)(a_img[index+1](r, c)), (uint8_t)(a_img[index+2](r, c)));
            dlib::assign_pixel(img(r, c),p);
        }
    }

}   // end of merge_channels

// ----------------------------------------------------------------------------------------

template<typename pixel_type>
void split_channels(dlib::matrix<dlib::rgb_pixel> img, uint64_t index, std::array<dlib::matrix<pixel_type>, img_depth> &img_s)
{
    uint64_t idx, r, c;
    uint32_t channels = 0;
    // get the size of the image
    uint64_t rows = img.nr();
    uint64_t cols = img.nc();

    //if (dlib::pixel_traits<pixel_type1>::rgb_alpha == true)
    //{
    //    channels = 4;
    //}
    //else if (dlib::pixel_traits<pixel_type1>::grayscale == true)
    //{
    //    channels = 1;
    //}
    //else
    //{
    //    channels = 3;
    //}

    //channels = dlib::pixel_traits<pixel_type1>::num;

    DLIB_CASSERT(img_s.size() >= channels, "not the right size");


    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            dlib::rgb_pixel p;
            dlib::assign_pixel(p, img(r, c));
            dlib::assign_pixel(img_s[index+0](r, c), p.red);
            dlib::assign_pixel(img_s[index+1](r, c), p.green);
            dlib::assign_pixel(img_s[index+2](r, c), p.blue);
            //dlib::assign_pixel(p, d(r, c));
            //dlib::assign_pixel(t[3](r, c), p.red);
            //dlib::assign_pixel(t[4](r, c), p.green);
            //dlib::assign_pixel(t[5](r, c), p.blue);
        }
    }


   


}   // end of split_channels


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
    //std::array<dlib::matrix<uint16_t>, img_depth> tr_crop;

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

        // get the location of the network
        std::string net_name = "../nets/dfd_net_pso_01_03_HPC.dat";
        
        //declare the network
        dfd_net_type dfd_net;

        // deserialize the network
        //dlib::deserialize(net_name) >> dfd_net;

        std::cout << dfd_net << std::endl;

        bp = 1;

        // setup the input image info
        std::string data_directory = "D:/IUPUI/Test_Data/Middlebury_Images_Third/Aloe/Illum2/Exp1/";
        std::string f_img = "view1.png";
        std::string d_img = "view1_lin_0.32_2.88.png";

        // load the images
        std::array<dlib::matrix<uint16_t>, img_depth> t;
        dlib::matrix<dlib::rgb_pixel> f, f_tmp, d, d_tmp;

        dlib::load_image(f_tmp, (data_directory+f_img));
        dlib::load_image(d_tmp, (data_directory+d_img));

        //split_channels(f_tmp, t);

        // crop the images to the right network size
        // get image size
        long rows = 352;
        long cols = 416;

        f.set_size(rows, cols);
        d.set_size(rows, cols);
        
        // crop the image to fit into the net
        dlib::set_subm(f, 0, 0, rows, cols) = dlib::subm(f_tmp, 0, 0, rows, cols);
        dlib::set_subm(d, 0, 0, rows, cols) = dlib::subm(d_tmp, 0, 0, rows, cols);
        
        // get the images size and resize the t array
        for (int m = 0; m < img_depth; ++m)
        {
            t[m].set_size(f.nr(), f.nc());
        }

        for (long r = 0; r < f.nr(); ++r)
        {
            for (long c = 0; c < f.nc(); ++c)
            {
                dlib::rgb_pixel p;
                dlib::assign_pixel(p, f(r, c));
                dlib::assign_pixel(t[0](r, c), p.red);
                dlib::assign_pixel(t[1](r, c), p.green);
                dlib::assign_pixel(t[2](r, c), p.blue);
                dlib::assign_pixel(p, d(r, c));
                dlib::assign_pixel(t[3](r, c), p.red);
                dlib::assign_pixel(t[4](r, c), p.green);
                dlib::assign_pixel(t[5](r, c), p.blue);
            }
        }

        // run the image through the network
        dlib::matrix<uint16_t> map = dfd_net(t);

        bp = 2;

        dlib::matrix < dlib::rgb_pixel> t3;
        merge_channels(t, 0, t3);

        // start looking at how to view the inards
        const auto& test = dlib::layer<0>(dfd_net).loss_details();
        //auto t2 = test.compute_loss_value_and_gradient();
        //const auto *t2 = test.to_label();

        bp = 3;
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
//        std::cout << "Press enter to close the program." << std::endl;
//        std::cin.ignore();

    }

    std::cout << "Press Enter to close" << std::endl;
    std::cin.ignore();
	return 0;

}	// end of main

