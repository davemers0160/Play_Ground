#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
//#include <windows.h>
//#include "win_network_fcns.h"
//#include <winsock2.h>
//#include <iphlpapi.h>
//
//#pragma comment(lib, "IPHLPAPI.lib")    // Link with Iphlpapi.lib

#endif



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
#include <type_traits>

// dlib includes
#include "dlib/rand.h"
#include "dlib/matrix.h"
#include "dlib/pixel.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "dlib/opencv.h"

// OpenCV includes
#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/video/video.hpp>

// custom includes
//#include "mmaplib.h"
#include "pg.h"
//#include "mmap.h"
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "read_binary_image.h" 
#include "make_dir.h"
#include "ssim.h"
#include "dlib_matrix_threshold.h"
#include "gorgon_capture.h"
#include "modulo.h"

//#include "pso.h"
//#include "ycrcb_pixel.h"
//#include "dfd_array_cropper.h"
#include "rot_90.h"
//#include "dlib_srelu.h"
//#include "dlib_elu.h"
#include "center_cropper.h"
#include "dfd_cropper_rw.h"

// new copy and set learning rate includes
#include "copy_dlib_net.h"
#include "dlib_set_learning_rates.h"

// Network includes
//#include "dfd_net_v14.h"
//#include "dfd_net_v14_pso_01.h"
//#include "dfd_net_rw_v19.h"
//#include "load_dfd_rw_data.h"
//#include "load_dfd_data.h"

//#include "cyclic_analysis.h"

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth = 3;
extern const uint32_t array_depth = 3;
extern const uint32_t secondary;

//extern const std::vector<std::pair<uint64_t, uint64_t>> crop_sizes;
std::string platform;
std::vector<std::array<dlib::matrix<uint16_t>, img_depth>> tr, te, trn_crop, te_crop;
//std::vector<dlib::matrix<uint16_t>> gt_train, gt_test, gt_crop, gt_te_crop;

//std::string version;
std::string net_name = "dfd_net_";
std::string net_sync_name = "dfd_sync_";
std::string logfileName = "dfd_net_";
std::string gorgon_savefile = "gorgon_dfd_";

dlib::rand rnd(time(NULL));

//using mnist_net_type = dlib::loss_multiclass_log<
//    dlib::fc<10,
//    dlib::prelu<dlib::fc<84,
//    dlib::prelu<dlib::fc<120,
//    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<16, 5, 5, 1, 1,
//    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<6, 5, 5, 1, 1,
//    dlib::input<dlib::matrix<unsigned char>>
//    >>>>>>>>>>>>;

//using mnist_net_type = dlib::loss_multiclass_log<
//    dlib::fc<10,
//    dlib::htan<dlib::fc<84,
//    dlib::sig<dlib::con<120, 5, 5, 1, 1,
//    dlib::sig<dlib::max_pool<2, 2, 2, 2, dlib::con<16, 5, 5, 1, 1,
//    dlib::sig<dlib::max_pool<2, 2, 2, 2, dlib::con<6, 5, 5, 1, 1,
//    dlib::input<dlib::matrix<unsigned char>>
//    >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
/*

template <typename T>
const dlib::matrix<T> or(
    const dlib::matrix<T>& m1,
    const dlib::matrix<T>& m2
    )
{

    dlib::matrix<T> result = dlib::zeros_matrix<T>(m1.nr(), m1.nc());
    for (uint64_t r = 0; r < m1.nr(); ++r)
    {
        for (uint64_t c = 0; c < m1.nc(); ++c)
        {
            result(r, c) = (m1(r, c) > 0) | (m2(r, c) > 0);
        }
    }
        return result;
}


// ----------------------------------------------------------------------------------------
template <typename T>
const dlib::matrix<T> and(
    const dlib::matrix<T>& m1,
    const dlib::matrix<T>& m2
    )
{

    dlib::matrix<T> result = dlib::zeros_matrix<T>(m1.nr(), m1.nc());
    for (uint64_t r = 0; r < m1.nr(); ++r)
    {
        for (uint64_t c = 0; c < m1.nc(); ++c)
        {
            result(r, c) = (m1(r, c) > 0) & (m2(r, c) > 0);
        }
    }
    return result;
}
*/

// ----------------------------------------------------------------------------------------

template <typename img_type1>
void check_matrix(img_type1 img)
{
    if (dlib::is_matrix<img_type1>::value == true)
        std::cout << "matrix" << std::endl;
}

// ----------------------------------------------------------------------------------------
/*
template <typename array_type>
void calc_linkdist(array_type in, dlib::matrix<uint16_t> &d)
{
    uint64_t idx, jdx;

    // assume that the vectors are in the rows
    // i.e. each row in the matrix represents a particular input

    dlib::matrix<uint16_t> one = dlib::ones_matrix<uint16_t>(in.nr(), in.nr());
    dlib::matrix<uint16_t> links = dlib::zeros_matrix<uint16_t>(in.nr(), in.nr());
    dlib::matrix<uint16_t> found = dlib::identity_matrix<uint16_t>(in.nr());
    dlib::matrix<uint16_t> next = dlib::zeros_matrix<uint16_t>(in.nr(), in.nr());
    dlib::matrix<uint16_t> newfound = dlib::zeros_matrix<uint16_t>(in.nr(), in.nr());

    d.set_size(in.nr(), in.nr());
    dlib::set_all_elements(d, in.nr());

    d = d - in.nr()*found;

    double s = 0.0;

    // get the links
    for (idx = 0; idx < in.nr(); ++idx)
    {
        for (jdx = 0; jdx < in.nr(); ++jdx)
        {
            if (idx != jdx)
            {
                s = std::sqrt((double)(dlib::sum(dlib::squared(dlib::rowm(in, idx) - dlib::rowm(in, jdx)))));

                //sum(idx, jdx) = s;

                if (s <= 1)
                    links(idx, jdx) = 1;
            }
        }
    }


    // cycle through the inputs by row
    for (idx = 0; idx < in.nr(); ++idx)
    {
        //nextfound = (found*links) | found;
        next = dlib::matrix_cast<uint16_t>((found*links));
        next = or(next, found);

        //newfound = nextfound & ~found;
        newfound = (one - found);
        newfound = and(next, newfound);

        for (uint64_t r = 0; r < in.nr(); ++r)
        {
            for (uint64_t c = 0; c < in.nr(); ++c)
            {
                if (newfound(r, c) == 1)
                    d(r, c) = idx + 1;
            }
        }

        found = next;

    }

}

*/

// ----------------------------------------------------------------------------------------

template <typename image_type1>
void make_random_cropping_rect(const image_type1& img, dlib::rectangle &rect_im, dlib::chip_dims dims= dlib::chip_dims(32,32))
{
    uint64_t x = 0, y = 0;

    rect_im = dlib::resize_rect(rect_im, dims.cols, dims.rows);
//    rect_gt = dlib::resize_rect(rect_gt, (long)(dims.cols / (double)scale_x), (long)(dims.rows / (double)scale_y));

    if ((unsigned long)img.nc() <= rect_im.width())
        x = 0;
    else
        x = (uint64_t)(rnd.get_integer(img.nc() - rect_im.width()));

    if ((unsigned long)img.nr() <= rect_im.height())
        y = 0;
    else
        y = (uint64_t)(rnd.get_integer(img.nr() - rect_im.height()));

    // randomly shift the box around
    dlib::point tr_off(x, y);
    rect_im = dlib::move_rect(rect_im, tr_off);

    //dlib::point gt_off(x, y);
    //rect_gt = dlib::move_rect(rect_gt, gt_off);


}	// end of make_random_cropping_rect 

// ----------------------------------------------------------------------------------------

template <typename img_type, typename T>
void create_mask(img_type src, img_type &mask, T min_value, T max_value)
{
    uint64_t nr = src.nr();
    uint64_t nc = src.nc();

    mask.set_size(nr, nc);

    for (uint64_t r = 0; r < nr; ++r)
    {
        for (uint64_t c = 0; c < nc; ++c)
        {
            if ((src(r, c) >= min_value) && (src(r,c) <= max_value))
                mask(r, c) = 1;
            else
                mask(r, c) = 0;
        }
    }

}   // end of create_mask

// ----------------------------------------------------------------------------------------

template <typename img_type, typename mask_type, typename T>
void apply_mask(img_type src, img_type &dst, mask_type mask, T value)
{
    uint64_t nr, nc;

    //if (dlib::is_matrix<img_type>::value == true)
    //{
    //    nr = src.nr();
    //    nc = src.nc();

    //    dst.set_size(nr, nc);

    //    for (uint64_t r = 0; r < nr; ++r)
    //    {
    //        for (uint64_t c = 0; c < nc; ++c)
    //        {
    //            if (mask(r, c) == 0)
    //                dst(r, c) = value;
    //            else
    //                dst(r, c) = src(r, c);
    //        }
    //    }

    //}
    //else
    //{
        nr = src[0].nr();
        nc = src[0].nc();

        for (uint64_t idx = 0; idx < src.size(); ++idx)
        {
            dst[idx].set_size(nr, nc);

            for (uint64_t r = 0; r < nr; ++r)
            {
                for (uint64_t c = 0; c < nc; ++c)
                {
                    if (mask(r, c) == 0)
                        dst[idx](r, c) = value;
                    else
                        dst[idx](r, c) = src[idx](r, c);
                }
            }
        }

    //}

}   // end of apply_mask

// ----------------------------------------------------------------------------------------

// This block of statements defines the resnet-34 network
template <long num_filters, typename SUBNET> using con3 = dlib::con<num_filters, 3, 3, 1, 1, SUBNET>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N1, int N2, typename SUBNET> using res33 = con3<N1, dlib::prelu<dlib::bn_con<con3<N2, SUBNET>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using level1 = ares<512, ares<512, ares_down<512, SUBNET>>>;
template <typename SUBNET> using level2 = ares<256, ares<256, ares<256, ares<256, ares<256, ares_down<256, SUBNET>>>>>>;
template <typename SUBNET> using level3 = ares<128, ares<128, ares<128, ares_down<128, SUBNET>>>>;
template <typename SUBNET> using level4 = ares<64, ares<64, ares<64, SUBNET>>>;

// This is the original network definition for the pretrained network dnn file
using anet_type = dlib::loss_multiclass_log< dlib::fc<1000, dlib::avg_pool_everything<
    level1<
    level2<
    level3<
    level4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    dlib::input_rgb_image_sized<227>
    >>>>>>>>>>>;

// This is the new net that you want to copy
using anet_type2 = dlib::loss_mmod<dlib::con<1,9,9,1,1,
    res33<512, 512,
    // ----
    level1<
    level2<
    level3<
    level4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    // ----
    dlib::input<std::array<dlib::matrix<uint8_t>, array_depth>>
    >>>>>>>>>>>;

// ----------------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx=0, jdx=0;

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    unsigned long training_duration = 1;  // number of hours to train 
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::vector<double> stop_criteria;
    uint64_t num_crops = 0;
    //std::vector<std::pair<uint64_t, uint64_t>> crop_sizes = { {1,1}, {38,148} };
    //std::vector<uint32_t> filter_num;
    //uint64_t max_one_step_count;
    std::vector<dlib::matrix<unsigned char>> training_images;
    std::vector<dlib::matrix<unsigned char>> testing_images;
    std::vector<unsigned long> training_labels;
    std::vector<unsigned long> testing_labels;

    std::vector<std::vector<std::string>> training_file;
    std::vector<std::vector<std::string>> test_file;

    std::string data_directory;
    std::string train_inputfile, test_inputfile;
    std::vector<std::pair<std::string, std::string>> tr_image_files, te_image_files;
    std::vector<dlib::matrix<uint16_t>> gt_train, gt_test;

    dlib::matrix<uint16_t> g_crop;
    dlib::rectangle rect_im, rect_gt;

    //std::array<dlib::matrix<uint16_t>, img_depth> tr_crop;

    std::ofstream DataLogStream;
    std::string platform;

    get_platform(platform);
    std::cout << "Platform: " << platform << std::endl;

    if (platform.compare(0, 6, "Laptop") == 0)
    {
        std::cout << "Match!" << std::endl;
    }

    try
    {
        int bp = 0;

        #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
            
        #else
            std::string exe_path = get_linux_path();
            std::cout << "Path: " << exe_path << std::endl;
        #endif        

        anet_type2 net_34_2;

        std::vector<std::string> labels;
        anet_type net_34;
        dlib::deserialize("../nets/resnet34_1000_imagenet_classifier.dnn") >> net_34 >> labels;

        // print out the networks to get the layer numbers for future copying
        std::cout << "net 34" << std::endl;
        std::cout << net_34 << std::endl;
      
        std::cout << "net 34 v2" << std::endl;
        std::cout << net_34_2 << std::endl;

        bp = 2;

        // This is how to copy a network:
        // dlib::copy_net<start_from, end_from, start_to>(from_net, to_net);
        dlib::copy_net<3, 143, 6>(net_34, net_34_2);

        // This is a check of a few layers to show that the nets are the same
        auto& tmp1 = dlib::layer<138>(net_34).layer_details().get_layer_params();
        auto& tmp2 = dlib::layer<137>(net_34_2).layer_details().get_layer_params();

        auto& tmp3 = dlib::layer<140>(net_34).layer_details().get_layer_params();
        auto& tmp4 = dlib::layer<139>(net_34_2).layer_details().get_layer_params();

        auto& tmp5 = dlib::layer<141>(net_34).layer_details().get_layer_params();
        auto& tmp6 = dlib::layer<140>(net_34_2).layer_details().get_layer_params();

        auto& tmp7 = dlib::layer<142>(net_34).layer_details();
        auto& tmp8 = dlib::layer<141>(net_34_2).layer_details();

        //auto& tmp9 = dlib::layer<143>(net_34).layer_details().get_layer_params();
        //auto& tmp10 = dlib::layer<142>(net_34_2).layer_details().get_layer_params();
        
        bp = 3;

        // set the learning rate multipliers: 0 means freeze the layers
        double r1 = 0.0, r2 = 0.0;
        dlib::set_learning_rate<6, 145>(net_34_2, r1, r2);

        // print out the net just to show that the multipliers have changed
        std::cout << "net 34 v2" << std::endl;
        std::cout << net_34_2 << std::endl;

        bp = 4;


    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    std::cout << "Press Enter to close" << std::endl;
    std::cin.ignore();

	return 0;

}	// end of main

