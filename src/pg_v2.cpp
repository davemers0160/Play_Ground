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
#include "gorgon_capture.h"

//#include "pso.h"
//#include "ycrcb_pixel.h"
//#include "dfd_array_cropper.h"
#include "rot_90.h"
//#include "dlib_srelu.h"
//#include "dlib_elu.h"
#include "center_cropper.h"
#include "dfd_cropper_rw.h"

// Network includes
#include "dfd_net_v14.h"
//#include "dfd_net_v14_pso_01.h"
//#include "dfd_net_rw_v19.h"
#include "load_dfd_rw_data.h"

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
extern const uint32_t secondary;
//extern const std::vector<std::pair<uint64_t, uint64_t>> crop_sizes;
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

template<typename array_type>
void merge_channels(array_type &a_img, uint64_t index, dlib::matrix<dlib::rgb_pixel> &img)
{
    uint64_t r, c;
    uint32_t channels = dlib::pixel_traits<dlib::rgb_pixel>::num;

    DLIB_CASSERT(a_img.size() >= (channels + index), "Array image does not contain enough channels: "
        << "Array Size: " << a_img.size() << "; index + channels: " << (channels + index));

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

template<typename array_type>
//void split_channels(dlib::matrix<dlib::rgb_pixel> img, uint64_t index, std::array<dlib::matrix<pixel_type>, img_depth> &img_s)
void split_channels(dlib::matrix<dlib::rgb_pixel> &img, uint64_t index, array_type &img_s)
{
    uint64_t idx, r, c;
    uint32_t channels = dlib::pixel_traits<dlib::rgb_pixel>::num;

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

    DLIB_CASSERT(img_s.size() >= (channels+index), "Array image does not contain enough channels:"
        << "Array Size: " << img_s.size() << "; index + channels: " << (channels + index));

    // set the size of the image
    img_s[index + 0].set_size(rows, cols);
    img_s[index + 1].set_size(rows, cols);
    img_s[index + 2].set_size(rows, cols);

    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            dlib::rgb_pixel p;
            dlib::assign_pixel(p, img(r, c));
            dlib::assign_pixel(img_s[index+0](r, c), p.red);
            dlib::assign_pixel(img_s[index+1](r, c), p.green);
            dlib::assign_pixel(img_s[index+2](r, c), p.blue);
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
    //std::vector<std::pair<uint64_t, uint64_t>> crop_sizes = { {1,1}, {38,148} };
    //std::vector<uint32_t> filter_num;
    //uint64_t max_one_step_count;

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
        std::string net_name = "D:/IUPUI/PhD/Results/dfd_dnn/2641623.pbs01_0_nets/nets/dfd_net_v14b_61_U_32_HPC.dat";
        
        //declare the network
        dfd_net_type dfd_net;

        // deserialize the network
        dlib::deserialize(net_name) >> dfd_net;

        std::cout << dfd_net << std::endl;

        bp = 1;

        // setup the input image info
        std::string data_directory = "D:/IUPUI/Test_Data/Middlebury_Images_Third/Art/Illum2/Exp1/";
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
        uint64_t rows = 368;// crop_sizes[1].first;
        uint64_t cols = 400;// crop_sizes[1].second;

        f.set_size(rows, cols);
        d.set_size(rows, cols);
        
        // crop the image to fit into the net
        dlib::set_subm(f, 0, 0, rows, cols) = dlib::subm(f_tmp, 0, 0, rows, cols);
        dlib::set_subm(d, 0, 0, rows, cols) = dlib::subm(d_tmp, 0, 0, rows, cols);
        
        // get the images size and resize the t array
        //for (int m = 0; m < img_depth; ++m)
        //{
        //    t[m].set_size(f.nr(), f.nc());
        //}

        //for (long r = 0; r < f.nr(); ++r)
        //{
        //    for (long c = 0; c < f.nc(); ++c)
        //    {
        //        dlib::rgb_pixel p;
        //        dlib::assign_pixel(p, f(r, c));
        //        dlib::assign_pixel(t[0](r, c), p.red);
        //        dlib::assign_pixel(t[1](r, c), p.green);
        //        dlib::assign_pixel(t[2](r, c), p.blue);
        //        dlib::assign_pixel(p, d(r, c));
        //        dlib::assign_pixel(t[3](r, c), p.red);
        //        dlib::assign_pixel(t[4](r, c), p.green);
        //        dlib::assign_pixel(t[5](r, c), p.blue);
        //    }
        //}

        split_channels(f, 0, t);
        split_channels(d, 3, t);

        // run the image through the network
        dlib::matrix<uint16_t> map = dfd_net(t);

        bp = 2;

        //dlib::matrix < dlib::rgb_pixel> t3;
        //merge_channels(t, 0, t3);

        // start looking at how to view the inards
        //const auto& test = dlib::layer<50>(dfd_net).get_output();
        //const float *t2 = test.host();
        //uint64_t n = test.num_samples();
        //uint64_t k = test.k();
        //uint64_t nr = test.nr();
        //uint64_t nc = test.nc();
        //uint64_t img_size = nr * nc;
        //uint64_t offset = 0;

        //dlib::matrix<float> o_img(nr, nc);
        //uint64_t index = 0;

        //for (uint64_t r = 0; r < nr; ++r)
        //{
        //    for (uint64_t c = 0; c < nc; ++c)
        //    {
        //        o_img(r, c) = *(t2 + (offset*img_size) + index);
        //        ++index;
        //    }
        //}

        std::string save_location = "../results/dfd_net_v14b/dfd_v14b_output_";

        gorgon_capture<50> gc_01(dfd_net);
        gc_01.init((save_location + "art_L50"));
        gc_01.save_net_output(dfd_net);
        gc_01.close_stream();

        gorgon_capture<46> gc_02(dfd_net);
        gc_02.init((save_location + "art_L46"));
        gc_02.save_net_output(dfd_net);
        gc_02.close_stream();

        gorgon_capture<45> gc_03(dfd_net);
        gc_03.init((save_location + "art_L45"));
        gc_03.save_net_output(dfd_net);
        gc_03.close_stream();

        gorgon_capture<42> gc_04(dfd_net);
        gc_04.init((save_location + "art_L42"));
        gc_04.save_net_output(dfd_net);
        gc_04.close_stream();

        gorgon_capture<38> gc_05(dfd_net);
        gc_05.init((save_location + "art_L38"));
        gc_05.save_net_output(dfd_net);
        gc_05.close_stream();

        gorgon_capture<37> gc_06(dfd_net);
        gc_06.init((save_location + "art_L37"));
        gc_06.save_net_output(dfd_net);
        gc_06.close_stream();

        gorgon_capture<34> gc_07(dfd_net);
        gc_07.init((save_location + "art_L34"));
        gc_07.save_net_output(dfd_net);
        gc_07.close_stream();

        gorgon_capture<30> gc_08(dfd_net);
        gc_08.init((save_location + "art_L30"));
        gc_08.save_net_output(dfd_net);
        gc_08.close_stream();

        gorgon_capture<29> gc_09(dfd_net);
        gc_09.init((save_location + "art_L29"));
        gc_09.save_net_output(dfd_net);
        gc_09.close_stream();

        gorgon_capture<27> gc_10(dfd_net);
        gc_10.init((save_location + "art_L27"));
        gc_10.save_net_output(dfd_net);
        gc_10.close_stream();

        gorgon_capture<22> gc_11(dfd_net);
        gc_11.init((save_location + "art_L22"));
        gc_11.save_net_output(dfd_net);
        gc_11.close_stream();

        gorgon_capture<18> gc_12(dfd_net);
        gc_12.init((save_location + "art_L18"));
        gc_12.save_net_output(dfd_net);
        gc_12.close_stream();

        gorgon_capture<17> gc_13(dfd_net);
        gc_13.init((save_location + "art_L17"));
        gc_13.save_net_output(dfd_net);
        gc_13.close_stream();

        gorgon_capture<10> gc_14(dfd_net);
        gc_14.init((save_location + "art_L10"));
        gc_14.save_net_output(dfd_net);
        gc_14.close_stream();

        gorgon_capture<6> gc_15(dfd_net);
        gc_15.init((save_location + "art_L06"));
        gc_15.save_net_output(dfd_net);
        gc_15.close_stream();

        gorgon_capture<5> gc_16(dfd_net);
        gc_16.init((save_location + "art_L05"));
        gc_16.save_net_output(dfd_net);
        gc_16.close_stream();

        gorgon_capture<2> gc_17(dfd_net);
        gc_17.init((save_location + "art_L02"));
        gc_17.save_net_output(dfd_net);
        gc_17.close_stream();

        gorgon_capture<1> gc_18(dfd_net);
        gc_18.init((save_location + "art_L01"));
        gc_18.save_net_output(dfd_net);
        gc_18.close_stream();

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

