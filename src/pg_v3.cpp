#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

//#include "edtinc.h"
//#include "libedt.h"

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#include <windows.h>
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
#include <array>
#include <algorithm>
#include <type_traits>
#include <list>
#include <thread>
#include <complex>
#include <random>
#include <functional>


// dlib includes
#include "dlib/rand.h"
#include "dlib/matrix.h"
#include "dlib/pixel.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "dlib/opencv.h"
#include "dlib/gui_widgets.h"
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <dlib/numeric_constants.h>
#include <dlib/geometry.h>
#include <dlib/enable_if.h>

// OpenCV includes
#include <opencv2/core.hpp>           
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp> 
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>

// custom includes
//#include "mmaplib.h"
#include "pg.h"

//#include "../../playground/include/cv_random_image_gen.h"
//#include "../../dlib_object_detection/common/include/obj_det_net_rgb_v10.h"


#include "mmap.h"
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "read_binary_image.h" 
//#include "make_dir.h"
#include "file_ops.h"
//#include "ssim.h"
#include "dlib_matrix_threshold.h"
#include "gorgon_capture.h"
#include "modulo.h"
#include "dlib_overlay_bbox.h"
#include "simplex_noise.h"
#include "ocv_threshold_functions.h"
//#include "so_cam_commands.h"
#include "console_input.h"

#include <cv_create_gaussian_kernel.h>

// dsp
#include <dsp/dsp_windows.h>

//#include "pso.h"
//#include "pso_particle.h"
//#include "ycrcb_pixel.h"
//#include "dfd_array_cropper.h"
#include "rot_90.h"
//#include "dlib_srelu.h"
//#include "dlib_elu.h"
#include "center_cropper.h"
//#include "dfd_cropper.h"
#include "target_locator.h"

// new copy and set learning rate includes
#include "copy_dlib_net.h"
#include "dlib_set_learning_rates.h"

#include "dlib_pixel_operations.h"

// Network includes
//#include "dfd_net_v14.h"
//#include "dfd_net_v14_pso_01.h"
//#include "dfd_net_rw_v19.h"
//#include "load_dfd_rw_data.h"
//#include "load_dfd_data.h"
//#include "resnet101_v1.h"

//#include "cyclic_analysis.h"

//#include <yaml-cpp/yaml.h>

#define M_PI 3.14159265358979323846
#define M_2PI 6.283185307179586476925286766559

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
extern const uint32_t array_depth=3;
extern const uint32_t secondary;

//extern const std::vector<std::pair<uint64_t, uint64_t>> crop_sizes;
std::string platform;
//std::vector<std::array<dlib::matrix<uint16_t>, img_depth>> tr, te, trn_crop, te_crop;
//std::vector<dlib::matrix<uint16_t>> gt_train, gt_test, gt_crop, gt_te_crop;

//std::string version;
std::string net_name = "dfd_net_";
std::string net_sync_name = "dfd_sync_";
std::string logfileName = "dfd_net_";
std::string gorgon_savefile = "gorgon_dfd_";

dlib::rand rnd(time(NULL));

int thresh = 50, N = 11;
const char* window_name = "Test";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = (double)pt1.x - (double)pt0.x;
    double dy1 = (double)pt1.y - (double)pt0.y;
    double dx2 = (double)pt2.x - (double)pt0.x;
    double dy2 = (double)pt2.y - (double)pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
static void findSquares(const cv::Mat& image, vector<vector<cv::Point> >& squares)
{
    squares.clear();

    cv::Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    cv::pyrDown(image, pyr, cv::Size(image.cols / 2, image.rows / 2));
    cv::pyrUp(pyr, timg, image.size());
    vector<vector<cv::Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
        int ch[] = { c, 0 };
        cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for (int l = 0; l < N; l++)
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                cv::Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l + 1) * 255 / N;
            }

            // find contours and store them all as a list
            cv::findContours(gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            vector<cv::Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++)
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(approx)) > 1000 &&
                    isContourConvex(approx))
                {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++)
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = std::abs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares(cv::Mat& image, const vector<vector<cv::Point> >& squares)
{
    for (size_t i = 0; i < squares.size(); i++)
    {
        const cv::Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        cv::polylines(image, &p, &n, 1, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }

    cv::imshow("test", image);
}


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
/*
// This is the original network definition for the pretrained network dnn file
using anet_type = dlib::loss_multiclass_log< dlib::fc<1000, dlib::avg_pool_everything<
    level1<
    level2<
    level3<
    level4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    dlib::input_rgb_image_sized<227>
    >>>>>>>>>>>;
*/
// ----------------------------------------------------------------------------------------

using car_net_type = dlib::loss_mean_squared_multioutput<dlib::htan<dlib::fc<2,
    dlib::multiply<dlib::fc<5,
    dlib::multiply<dlib::fc<10,
    dlib::multiply<dlib::fc<50,
    dlib::input<dlib::matrix<float>>
    >> >> >> >>>;

car_net_type c_net(dlib::multiply_(0.001), dlib::multiply_(0.5), dlib::multiply_(0.5));
dlib::image_window win;
dlib::matrix<dlib::rgb_pixel> color_map;

//const uint64_t fc4_size = (1081 + 1) * 200;
//const uint64_t fc3_size = (200 + 1) * 50;
//const uint64_t fc2_size = (50 + 1) * 20;
//const uint64_t fc1_size = (20 + 1) * 2;
extern const uint64_t fc4_size;
extern const uint64_t fc3_size;
extern const uint64_t fc2_size;
extern const uint64_t fc1_size;

// This is the new net that you want to copy
//using anet_type2 = dlib::loss_mmod<dlib::con<1,9,9,1,1,
//    res33<512, 512,
//    // ----
//    level1<
//    level2<
//    level3<
//    level4<
//    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
//    // ----
//    dlib::input<std::array<dlib::matrix<uint8_t>, array_depth>>
//    >>>>>>>>>>>;

// ----------------------------------------------------------------------------------------	
double schwefel2(double x0, double x1, double x2, double x3)
{
    double result = 0.0;
    result += -x0 * std::sin(std::sqrt(std::abs(x0)));
    result += -x1 * std::sin(std::sqrt(std::abs(x1)));
    result += -x2 * std::sin(std::sqrt(std::abs(x2)));
    result += -x3 * std::sin(std::sqrt(std::abs(x3)));

    return result;
}

/*
double schwefel(particle p)
{
    // f(x_n) = -418.982887272433799807913601398*n = -837.965774544867599615827202796
    // x_n = 420.968746

    dlib::matrix<double> x1 = p.get_x1();
    dlib::matrix<double> x2 = p.get_x2();
    //double result = 418.9829* x.nc();

    //for (int64_t c = 0; c < x.nc(); ++c)
    //{
    //    result += -x(0, c) * std::sin(std::sqrt(std::abs(x(0, c))));
    //}

    //return result;

    //return schwefel2(x(0, 0), x(0, 1), x(0, 2), x(0, 3));
    return schwefel2(x1(0, 0), x1(0, 1), x2(0, 0), x2(0, 1));

}	// end of schwefel
*/
// ----------------------------------------------------------------------------------------

cv::Mat octave_image;

const int scale_slider_max = 100;
int scale_slider = 5;

const int octave_slider_max = 200;
int octave_slider = 3;

const int per_slider_max = 100;
int per_slider = 79;

std::vector<cv::Vec3b> color = { cv::Vec3b(41,44,35), cv::Vec3b(57,91,61), cv::Vec3b(80,114,113), cv::Vec3b(64,126,132) };

open_simplex_noise sn;


static void on_trackbar(int, void*)
{
    double v;
    uint32_t r, c;
    double scale, p;

    //scale = 1.0 / (double)(scale_slider + 1);
//    scale = ((0.01 - 0.005) / 20.0) * scale_slider + 0.001;
    scale = 0.00595 * scale_slider + 0.005;

    p = (double)(per_slider + 1) / 100.0;

    int o = max(1, octave_slider);

    std::cout << o << "," << p << "," << scale << std::endl;

    for (r = 0; r < octave_image.rows; ++r)
    {
        for (c = 0; c < octave_image.cols; ++c)
        {
            v = sn.octave((double)r * scale, (double)c * scale, o, p*1.0);
            uint8_t index = (uint8_t)(((v + 1.0) / 2.0) * 20);
            if(index<8)
                octave_image.at<cv::Vec3b>(r, c) = color[0];
            else if(index >= 8 && index < 10)
                octave_image.at<cv::Vec3b>(r, c) = color[1];
            else if (index >= 10 && index < 12)
                octave_image.at<cv::Vec3b>(r, c) = color[2];
            else
                octave_image.at<cv::Vec3b>(r, c) = color[3];

        }
    }


    cv::imshow(window_name, octave_image);
}

// ----------------------------------------------------------------------------------------
template <typename T>
double nan_mean(cv::Mat& img)
{
    uint64_t count = 0;
    double mn = 0;

    cv::MatIterator_<T> it;

    for (it = img.begin<T>(); it != img.end<T>(); ++it)
    {
        if (!std::isnan(*it))
        {
            mn += (double)*it;
            ++count;
        }
    }

    return (mn / (double)count);
}   // end of nan_mean


dlib::matrix<uint32_t, 1, 4> get_color_match(dlib::matrix<dlib::rgb_pixel>& img, dlib::mmod_rect& det)
{
    uint64_t r, c;

    dlib::hsi_pixel red_ll1(0, 0, 30);
    dlib::hsi_pixel red_ul1(15, 255, 192);
    dlib::hsi_pixel red_ll2(240, 0, 30);
    dlib::hsi_pixel red_ul2(255, 255, 192);

    dlib::hsi_pixel blue_ll(150, 0, 30);
    dlib::hsi_pixel blue_ul(190, 255, 192);

    //dlib::hsi_pixel black_ll(0, 0, 0);
    //dlib::hsi_pixel black_ul(255, 64, 48);
    dlib::rgb_pixel black_ll(0, 0, 0);
    dlib::rgb_pixel black_ul(48, 48, 48);

    dlib::hsi_pixel green_ll(65, 0, 30);
    dlib::hsi_pixel green_ul(105, 255, 192);

    dlib::hsi_pixel gray_ll(0, 0, 48);
    dlib::hsi_pixel gray_ul(255, 255, 128);
    //dlib::rgb_pixel gray_ll(65, 65, 65);
    //dlib::rgb_pixel gray_ul(128, 128, 128);

    const int w = 20, h = 20;

    dlib::matrix<uint16_t> red_mask = dlib::zeros_matrix<uint16_t>(h, w);
    dlib::matrix<uint16_t> blue_mask = dlib::zeros_matrix<uint16_t>(h, w);
    dlib::matrix<uint16_t> black_mask = dlib::zeros_matrix<uint16_t>(h, w);
    dlib::matrix<uint16_t> gray_mask = dlib::zeros_matrix<uint16_t>(h, w);
    dlib::matrix<uint16_t> green_mask = dlib::zeros_matrix<uint16_t>(h, w);

    // crop out the detection
    dlib::point ctr = dlib::center(det.rect);

    dlib::matrix<dlib::rgb_pixel> rgb_crop = dlib::subm(img, dlib::centered_rect(ctr, w, h));
    dlib::matrix<dlib::hsi_pixel> hsi_crop;
    dlib::assign_image(hsi_crop, rgb_crop);

    dlib::hsi_pixel p;
    dlib::rgb_pixel q;

    for (r = 0; r < hsi_crop.nr(); ++r)
    {
        for (c = 0; c < hsi_crop.nc(); ++c)
        {
            dlib::assign_pixel(p, hsi_crop(r, c));
            dlib::assign_pixel(q, rgb_crop(r, c));

            // test for red backpack
            if ((p >= red_ll1) && (p <= red_ul1))
            {
                red_mask(r, c) = 1;
            }
            else if ((p >= red_ll2) && (p <= red_ul2))
            {
                red_mask(r, c) = 1;
            }
            else if ((p >= blue_ll) && (p <= blue_ul))
            {
                blue_mask(r, c) = 1;
            }
            else if ((p >= green_ll) && (p <= green_ul))
            {
                green_mask(r, c) = 1;
            }
            else if ((q >= black_ll) && (q <= black_ul))
            {
                black_mask(r, c) = 1;
            }
            else if ((p >= gray_ll) && (p <= gray_ul))
            {
                gray_mask(r, c) = 1;
            }

        }
    }

    uint32_t sum_cm = (uint32_t)dlib::sum(red_mask) + (uint32_t)dlib::sum(blue_mask) + (uint32_t)dlib::sum(black_mask);

    dlib::matrix<uint32_t, 1, 4> res;
    res = sum_cm, red_mask.size()-sum_cm, 0, 0;

    return res;

}


/*
This function will take a base image and overlay a set of shapes and then blur them according to the sigma value

@param src input image that will be modified and returned.  This image should be CV_32FC3 type
@param rng random number generator object
@param sigma value that determines the blur kernel properties
@param scale optional parameter to determine the size of the object
*/
void blur_layer(cv::Mat& src,
    cv::RNG& rng,
    double sigma,
    double scale = 0.3)
{

    // get the image dimensions for other uses
    int nr = src.rows;
    int nc = src.cols;

    // clone the source image
    cv::Mat src_clone = src.clone();

    // create the inital blank mask
    cv::Mat BM_1 = cv::Mat(src.size(), CV_32FC3, cv::Scalar::all(0.0));

    // generate a random color 
    cv::Scalar C = (1.0 / 255.0) * cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

    // generate a random point within the image using the cv::RNG uniform funtion (use nr,nc)
    int x = rng.uniform(0, nc);
    int y = rng.uniform(0, nr);

    // generate the random radius values for an ellipse using one of the RNG functions
    int r1 = std::floor(scale * rng.uniform(0, std::min(nr, nc)));
    int r2 = std::floor(scale * rng.uniform(0, std::min(nr, nc)));

    // generate a random angle between 0 and 360 degrees for the ellipse using one of the RNG functions
    double a = rng.uniform(0.0, 360.0);

    // use the cv::ellipse function and the random points to generate a random ellipse on top of the src_clone image
    cv::ellipse(src_clone, cv::Point(x, y), cv::Size(r1, r2), a, 0.0, 360.0, C, -1, cv::LineTypes::LINE_8, 0);

    // use the same points to generate the same ellipse on the mask with color = CV::Scalar(1.0, 1.0, 1.0)
    cv::ellipse(BM_1, cv::Point(x, y), cv::Size(r1, r2), a, 0.0, 360.0, cv::Scalar(1.0, 1.0, 1.0), -1, cv::LineTypes::LINE_8, 0);

    // blur the src_clone image with the overlay and blur the mask image using the sigma value to determine the blur kernel size
    cv::GaussianBlur(src_clone, src_clone, cv::Size(0, 0), sigma, sigma, cv::BORDER_REPLICATE);
    cv::GaussianBlur(BM_1, BM_1, cv::Size(0, 0), sigma, sigma, cv::BORDER_REPLICATE);

    // multiply the src image times (cv::Scalar(1.0, 1.0, 1.0) - BM_1)
    cv::Mat L1_1;
    cv::multiply(src, cv::Scalar(1.0, 1.0, 1.0) - BM_1, L1_1);

    // multiply the src_clone image times BM_1
    cv::Mat L1_2;
    cv::multiply(src_clone, BM_1, L1_2);

    // set src equal to L1_1 + L1_2
    src = L1_1 + L1_2;

}   // end of blur_layer 


// ----------------------------------------------------------------------------------------
template <typename in_image_type,typename out_image_type, typename hist_type>
void histogram_specification(in_image_type& in_img_, out_image_type& out_img_, hist_type &spec_hist)
{
    uint64_t idx, jdx;

    dlib::const_image_view<in_image_type> in_img(in_img_);
    dlib::image_view<out_image_type> out_img(out_img_);
    typedef typename dlib::image_traits<in_image_type>::pixel_type in_pixel_type;
    typedef typename dlib::image_traits<out_image_type>::pixel_type out_pixel_type;

    out_img.set_size(in_img.nr(), in_img.nc());

    unsigned long p;

    dlib::matrix<unsigned long, 1, 0> histogram;
    dlib::get_histogram(in_img_, histogram);
    in_img = in_img_;

    double scale = dlib::pixel_traits<out_pixel_type>::max();
    if (in_img.size() > histogram(0))
        scale /= in_img.size() - histogram(0);
    else
        scale = 0;

    // make the black pixels remain black in the output image
    histogram(0) = 0;

    // compute the transform function
    for (idx = 1; idx < histogram.size(); ++idx)
        histogram(idx) += histogram(idx - 1);
    // scale so that it is in the range [0,pixel_traits<out_pixel_type>::max()]
    for (idx = 0; idx < histogram.size(); ++idx)
        histogram(idx) = static_cast<unsigned long>(histogram(idx) * scale);


    dlib::matrix<unsigned char> map(1, histogram.size());
    double min_value;
    uint32_t index;
    for (idx = 0; idx < histogram.size(); ++idx)
    {
        min_value = 1000;
        index = histogram.size() - 1;
        for (jdx = 0; jdx < spec_hist.size(); ++jdx)
        {
            if (std::abs((double)histogram(idx) - (double)spec_hist(jdx)) < min_value)
            {
                min_value = std::abs((double)histogram(idx) - (double)spec_hist(jdx));
                index = jdx;
            }
        }
        map(idx) = index;
    }

    // now do the transform
    for (long row = 0; row < in_img.nr(); ++row)
    {
        for (long col = 0; col < in_img.nc(); ++col)
        {
            p = map(get_pixel_intensity(in_img[row][col]));
            assign_pixel(out_img[row][col], in_img[row][col]);
            assign_pixel_intensity(out_img[row][col], p);
        }
    }


    int bp = 0;
}

// ----------------------------------------------------------------------------
void generate_checkerboard(uint32_t block_w, uint32_t block_h, uint32_t img_w, uint32_t img_h, cv::Mat& checker_board)
{
    uint32_t idx = 0, jdx = 0;

    cv::Mat white = cv::Mat(block_w, block_h, CV_8UC3, cv::Scalar::all(255));

    checker_board = cv::Mat(img_h + block_h, img_w + block_w, CV_8UC3, cv::Scalar::all(0));

    bool color_row = false;
    bool color_column = false;

    for (idx = 0; idx < img_h; idx += block_h)
    {
        color_row = !color_row;
        color_column = color_row;

        for (jdx = 0; jdx < img_w; jdx += block_w)
        {
            if (!color_column)
                white.copyTo(checker_board(cv::Rect(jdx, idx, block_w, block_h)));

            color_column = !color_column;
        }

    }

    // need to add cropping of image
    cv::Rect rio(0, 0, img_w, img_h);
    checker_board = checker_board(rio);
}


void distortion(cv::Mat& src, cv::Mat& dst, double c_x, double c_y, double k_x1, double k_y1)
{
    long x, y;

    cv::Mat mapx = cv::Mat(src.size(), CV_32FC1, cv::Scalar::all(0.0));
    cv::Mat mapy = cv::Mat(src.size(), CV_32FC1, cv::Scalar::all(0.0));

    cv::Mat dst2 = cv::Mat(src.size(), CV_8UC3, cv::Scalar::all(0));

    int nc = src.cols;
    int nr = src.rows;
    uint64_t r = 0;
    uint64_t rr = 0;
    int64_t x_d, y_d;

    float xn, yn, xd_f, yd_f, x3, y3;

    for (y = 0; y < nr; ++y)
    {
        for (x = 0; x < nc; ++x)
        {
            xn = (float)(2 * x - nc) / (double)nc;
            yn = (float)(2 * y - nr) / (double)nr;

            r = ((x - c_x) * (x - c_x) + (y - c_y) * (y - c_y));
            rr = r * r;

            xd_f = xn * (1.0 + k_x1 * r);
            yd_f = yn * (1.0 + k_y1 * r);
            //xd_f = xn / (1.0 - k_x1 * (x3 * x3 + y3 * y3));
            //yd_f = yn / (1.0 - k_y1 * (x3 * x3 + y3 * y3));

            //xd_f = xn * (1 + k_x1 * r);// +k_x1 * 0.00001 * rr);
            //yd_f = yn * (1 + k_y1 * r);// +k_y1 * 0.00001 * rr);

            x_d = (int64_t)((xd_f + 1.0) * nc / 2.0);
            y_d = (int64_t)((yd_f + 1.0) * nr / 2.0);

            if(x_d >= 0 && x_d < nc && y_d >= 0 && y_d <  nr)
                dst2.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y_d, x_d);

            mapx.at<float>(y,x) = c_x + (x - c_x) * (1 + k_x1 * r);
            mapy.at<float>(y,x) = c_y + (y - c_y) * (1 + k_y1 * r);

        }
    }


    cv::remap(src, dst, mapx, mapy, cv::INTER_CUBIC);

    int bp = 0;
}



volatile bool entry = false;
volatile bool run = true;
std::string console_input1;

void get_input(void)
{
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (run == true)
        {
            std::getline(std::cin, console_input1);
            entry = true;
        }
        else
        {
            break;
        }
    }
}

//cv::Rect roi;

typedef struct mouse_params
{
    bool is_drawing = false;
    cv::Rect roi;

} mouse_params;


// ----------------------------------------------------------------------------------------
void mouse_click(int event, int x, int y, int flags, void* param)
{
    mouse_params* mp = (mouse_params*)param;

    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        (*mp).is_drawing = true;
        (*mp).roi = cv::Rect(x, y, 0, 0);
        break;

    case cv::EVENT_MOUSEMOVE:
        if ((*mp).is_drawing)
        {
            (*mp).roi = cv::Rect(std::min((*mp).roi.x, x), std::min((*mp).roi.y, y), std::abs(x - (*mp).roi.x), std::abs(y - (*mp).roi.y));
        }
        break;

    case cv::EVENT_LBUTTONUP:
        (*mp).roi = cv::Rect(std::min((*mp).roi.x, x), std::min((*mp).roi.y, y), std::abs(x - (*mp).roi.x), std::abs(y - (*mp).roi.y));
        (*mp).is_drawing = false;
        break;
    }

}   // end of mouse_click

// ----------------------------------------------------------------------------------------
static void meshgrid(const cv::Range& xgv, 
    const cv::Range& ygv,
    cv::Mat& X, 
    cv::Mat& Y)
{
    //const cv::Mat& xgv, const cv::Mat& ygv,
    //cv::Mat1i& X, cv::Mat1i& Y)

    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; ++i)
        t_x.push_back(i);

    for (int i = ygv.start; i <= ygv.end; ++i)
        t_y.push_back(i);

    cv::Mat x(t_x);
    cv::Mat y(t_y);

    cv::repeat(x.reshape(1, 1), y.total(), 1, X);
    cv::repeat(y.reshape(1, 1).t(), 1, x.total(), Y);
}

int test_pair(void* t)
{
    int bp2 = 0;

    //typedef u8_pair *std::pair<uint8_t, uint8_t>;


    auto t2 = (std::pair<uint8_t, uint8_t>*)t;

    std::vector<std::pair<uint8_t, uint8_t>> tp;

    tp.insert(tp.end(), &t2[0], &t2[3]);

    bp2 = 2;
    return bp2;
}



template <typename T>
struct fixed_width_mat
{
    fixed_width_mat(T m_, int w_) : m(m_), w(w_) {}
    T m;
    int w;
};

template<typename T>
//void print_matrix_as_csv(T &m, uint32_t w)
std::ostream& operator<< (std::ostream& out, const fixed_width_mat<T>& fmw)
{

    for (long r = 0; r < fmw.m.nr(); ++r)
    {
        for (long c = 0; c < fmw.m.nc(); ++c)
        {
            if (c + 1 == fmw.m.nc())
                out << std::setw(fmw.w) << fmw.m(r, c) << std::endl;
            else
                out << std::setw(fmw.w) << fmw.m(r, c) << ", ";
        }
    }

    return out;
}   // end of print_matrix_as_csv


// ----------------------------------------------------------------------------------------
namespace dlib
{
    //template <typename T>
    inline matrix<double> hann_window(int64_t N)
    {
        matrix<double> w = zeros_matrix<double>(N, 1);

        for (int64_t idx = 0; idx < N; ++idx)
        {
            w(idx, 0) = 0.5 * (1.0 - std::cos(2.0 * M_PI * idx / (double)(N - 1)));
        }

        return w;
    }   // end of hann_window

    inline matrix<double> blackman_nutall_window(int64_t N)
    {
        double a0 = 0.3635819;
        double a1 = 0.4891775;
        double a2 = 0.1365995;
        double a3 = 0.0106411;

        matrix<double> w = zeros_matrix<double>(1, N);

        for (int64_t idx = 0; idx < N; ++idx)
        {
            w(0, idx) = a0 - a1 * std::cos(2.0f * M_PI * idx / (double)(N - 1)) + a2 * std::cos(4.0f * M_PI * idx / (double)(N - 1)) - a3 * std::cos(6.0f * M_PI * idx / (double)(N - 1));
        }

        return w;
    }   // end of hann_window


    template <typename T, typename funct>
    matrix<T> create_fir_filter(int64_t N, float fc, funct window_function)
    {
        double v;
        matrix<float> g = zeros_matrix<float>(1, N);

        matrix<double> w = window_function(N);
        
        for (int64_t idx = 0; idx < N; ++idx)
        {
            if (std::abs((double)(idx - (N >> 1))) < 1e-6)
                g(0, idx) = w(0, idx) * fc;
            else
            {
                v = std::sin(M_PI * fc * (idx - ((N - 1) >> 1))) / (M_PI * (idx - ((N - 1) >> 1)));
                g(0, idx) = w(0, idx) * v;
            }
        }
        
        return matrix_cast<T>(g);

    }   // end of create_fir_filter

    //template <typename funct>
    dlib::matrix<std::complex<float>> create_fft_fir_filter(int64_t N, float fc, std::function<matrix<double>(int64_t)> window_function)
    {
        double v;
        matrix<std::complex<float>> g(1, N);
        //matrix<float> g = zeros_matrix<float>(1, N);


        matrix<double> w = window_function(N);

        for (int64_t idx = 0; idx < N; ++idx)
        {
            if (std::abs((double)(idx - (N >> 1))) < 1e-6)
                g(0, idx) = std::complex<float>(w(0, idx) * fc, 0.0f);
            else
            {
                v = std::sin(M_PI * fc * (idx - ((N - 1) >> 1))) / (M_PI * (idx - ((N - 1) >> 1)));
                g(0, idx) = std::complex<float>((float)w(0, idx) * v, 0.0f);
            }
        }

        fft_inplace(g);
        return g;

    }   // end of create_fir_filter

    template <typename M1, typename M2>
    inline bool type_test(M1 m1, M2 m2)
    {
        bool same_type = dlib::is_same_type<typename M1::type, typename M2::type>::value;

        return same_type;

    }

    template <typename M1,  typename funct>
    inline dlib::matrix<M1> apply_fir_filter(dlib::matrix<M1> &src, int64_t N, float fc, funct window_function)
    {
        dlib::matrix<M1> fir = create_fir_filter<M1>(N, fc, window_function);

        return dlib::conv_same(src, fir);
    }

    template <typename M1>
    inline dlib::matrix<M1> apply_fir_filter(dlib::matrix<M1> &src, dlib::matrix<M1> &filter)
    {
        return dlib::conv_same(src, fir);
    }


    template <typename T>
    dlib::matrix<std::complex<T>> generate_frequency_shift(uint64_t N, double f_offset, double fs)
    {
        uint64_t idx;

        dlib::matrix<std::complex<T>> dst = dlib::zeros_matrix<std::complex<T>>(N, 1);

        for (idx = 0; idx < N; ++idx)
        {
            dst(idx, 0) = (std::complex<T>)(std::exp(-2.0 * 1i * M_PI * (f_offset / (double)fs) * (double)idx));
        }

        return dst;
    }

    template <typename T>
    inline dlib::matrix<std::complex<T>> apply_frequency_shift(dlib::matrix<std::complex<T>>& src, double f_offset, double fs)
    {
        uint64_t idx;
        std::complex<T> v;

        dlib::matrix<std::complex<T>> dst = zeros_matrix(src);

        for (idx = 0; idx < src.size(); ++idx)
        {
            v = (std::complex<T>)(std::exp(-2.0 * 1i * M_PI * (f_offset / (double)fs) * (double)idx));
            dst(idx, 0) = src(idx, 0) * v;
        }

        return dst;
    }


    template <typename M1>
    inline dlib::matrix<M1> apply_frequency_shift(dlib::matrix<M1>& src, dlib::matrix<M1>& rot)
    {
        return pointwise_multiply(src, rot);
    }

}

// ----------------------------------------------------------------------------------------
constexpr size_t CARRIER = 2400;
constexpr size_t BAUD = 4160;
constexpr size_t OVERSAMPLE = 3;

template <typename F, typename T>
constexpr T map_value(F value, F f1, F t1, T f2, T t2) 
{
    return f2 + ((t2 - f2) * (value - f1)) / (t1 - f1);
}

std::vector<uint8_t> write_value(uint8_t value) 
{
    static double sn = 0;
    std::vector<uint8_t> buffer;

    for (size_t i = 0; i < OVERSAMPLE; i++) 
    {
        double samp = sin(CARRIER * 2.0 * M_PI * (sn / (BAUD * OVERSAMPLE)));
        samp *= map_value((int)value, 0, 255, 0.0, 0.7);

        uint8_t buf = map_value(samp, -1.0, 1.0, 0, 255);
        buffer.push_back(buf);

        //fwrite(buf, 1, 1, stdout);

        sn++;
    }

    return buffer;
}

typedef struct SignalDataPacket {
    uint32_t m_StreamIdentifier;
    uint64_t m_Timestamp_GetFractionalSeconds;

} SignalDataPacket;

template <typename T>
std::vector<T> find_intersection(std::vector<T> V1, std::vector<T> V2)
{
    uint64_t idx = 0, jdx = 0;
    std::vector<T> intersection;
    size_t n1 = V1.size();
    size_t n2 = V2.size();

    while (idx < n1 && jdx < n2)
    {
        if (V1[idx] > V2[jdx])
        {
            ++jdx;
        }
        else if (V2[jdx] > V1[idx])
        {
            ++idx;
        }
        else
        {
            intersection.push_back(V1[idx]);
            ++idx;
            ++jdx;
        }
    }

    return intersection;
}   // end of find_intersection

template <typename T>
size_t find_index(std::vector<T> vec, T value)
{
    size_t index = 0;

    while (vec[index] != value)
    {
        if (index == vec.size())
            return -1;

        ++index;
    }

    return index;
}


std::vector<uint64_t> find_match(std::vector<std::vector<uint64_t>> v)
{
    uint64_t idx;
    uint64_t N = v.size();
    std::vector<uint64_t> match(N);

    std::vector<uint64_t> intersection = find_intersection(v[0], v[1]);

    for (idx = 2; idx < N; ++idx)
    {
        intersection = find_intersection(intersection, v[idx]);
    }

    if (intersection.size() == 0)
        return match;

    // find the index of the first entry
    for (idx = 0; idx < N; ++idx)
    {
        match[idx] = find_index(v[idx], intersection[0]);
        std::cout << match[idx] << std::endl;
    }

    return match;
}


//-----------------------------------------------------------------------------
inline dlib::matrix<std::complex<float>> generate_oqpsk(std::vector<uint8_t> data, float amplitude, uint32_t sample_rate, float half_symbol_length)
{
    uint32_t idx = 0, jdx = 0;
    uint8_t num;

    uint32_t samples_per_bit = floor(sample_rate * half_symbol_length);
    uint32_t samples_per_symbol = samples_per_bit << 1;

    uint32_t i_index = 0, q_index = samples_per_bit;

    // check for odd numberand append a 0 at the end if it is odd
    if (data.size() % 2 == 1)
        data.push_back(0);

    uint32_t num_bit_pairs = data.size() >> 1;

    //% this will expand the bit to fill the right number of samples
    //s = ones(floor(2 * samples_per_bit), 1);

    // pre calculate the base 45 degree value
    float v = (float)(amplitude * 0.70710678118654752440084436210485);
    float v_i, v_q;

    //std::vector<std::complex<float>> iq_data(num_bit_pairs * samples_per_symbol + samples_per_bit);

    dlib::matrix<std::complex<float>> iq_data(1, num_bit_pairs * samples_per_symbol + samples_per_bit);

    // start with Iand Q offset by half a bit length
    //std::vector<int16_t> I;
    //std::vector<int16_t> Q(samples_per_bit, 0);

    std::vector<float> I(num_bit_pairs * samples_per_symbol + samples_per_bit, 0);
    std::vector<float> Q(num_bit_pairs * samples_per_symbol + samples_per_bit, 0);


    //v_i = data[idx] == 0 ? -v : v;
    //v_q = data[idx + 1] == 0 ? -v : v;

    //for (jdx = 0; jdx < samples_per_symbol; ++jdx)
    //{
        //I[i_index] = v_i;
        //Q[q_index] = v_q;
    //    ++i_index;
    //    ++q_index;
    //}


    for (idx = 0; idx < data.size(); idx += 2)
    {

        // map the bit pair value to IQ values
        v_i = data[idx] == 0 ? -v : v;
        v_q = data[idx + 1] == 0 ? -v : v;

        // append the new data
        //I.insert(I.end(), samples_per_symbol, v_i);
        //Q.insert(Q.end(), samples_per_symbol, v_q);

        for (jdx = 0; jdx < samples_per_symbol; ++jdx)
        {
            I[i_index] = v_i;
            Q[q_index] = v_q;
            ++i_index;
            ++q_index;
        }

    }

    // merge the Iand Q channels
    for (idx = 0; idx < I.size(); ++idx)
    {
        //iq_data.push_back(std::complex<int16_t>(I[idx], Q[idx]));
        iq_data(0, idx) = (std::complex<float>(I[idx], Q[idx]));
    }

    return iq_data;
}   // end of generate_oqpsk

void generate_channel_rot(uint32_t num_bits, uint64_t sample_rate, double half_symbol_length, std::vector<int64_t> &channels, std::vector<dlib::matrix<std::complex<float>>> &ch_rot)
{
    uint32_t idx, jdx;

    float ch;
    std::complex<float> j(0,1);

    if (num_bits % 2 == 1)
        ++num_bits;

    uint32_t num_bit_pairs = num_bits >> 1;
    uint32_t samples_per_bit = floor(sample_rate * half_symbol_length);

    uint32_t num_samples = num_bit_pairs * (samples_per_bit << 1) + samples_per_bit;

    ch_rot.clear();
    ch_rot.resize(channels.size());

    for (idx = 0; idx < channels.size(); ++idx)
    {
        ch_rot[idx] = dlib::matrix<std::complex<float>>(1, num_samples);

        ch = M_2PI * (channels[idx] / (float)sample_rate);

        for (jdx = 0; jdx < num_samples; ++jdx)
        {
            ch_rot[idx](0,jdx) = std::exp(j * (ch * jdx));
        }
    }

}   // end of generator_channel_rot

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx=0, jdx=0, kdx = 0;
    uint64_t r, c;
    char key = 0;

    typedef std::chrono::duration<double> d_sec;
    typedef chrono::nanoseconds ns;
    auto start_time = chrono::high_resolution_clock::now();
    auto stop_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<ns>(stop_time - start_time);

    unsigned long training_duration = 1;  // number of hours to train 

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
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

#else
        std::cout << "argv[0]: " << std::string(argv[0]) << std::endl;
        std::string exe_path = get_ubuntu_path();
        std::cout << "Path: " << exe_path << std::endl;
#endif

        int bp = 0;

        //std::vector<uint8_t> SYNCA = { 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        int64_t N = 64;
        int32_t bar_width = 100;

        uint32_t k;

        int32_t position = 0;

        double amplitude = 2000;
        uint32_t sample_rate = 50000000;
        double half_symbol_length = 0.0000001;

        uint32_t num_bits = 384;
        uint32_t num_bursts = 16;
        uint32_t ch, ch_rnd;

        std::vector<int64_t> channels = { -8000000, -7000000, -6000000, -5000000, -4000000, -3000000, -2000000, -1000000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000 };

        std::default_random_engine generator(time(0));

        std::uniform_int_distribution<int32_t> bits_gen = std::uniform_int_distribution<int32_t>(0, 1);
        std::uniform_int_distribution<int32_t> channel_gen = std::uniform_int_distribution<int32_t>(0, channels.size() - 1);

        std::vector<uint8_t> data(num_bits);

        std::vector<dlib::matrix<std::complex<float>>> ch_rot;
        //generate_channel_rot(num_bits, sample_rate, half_symbol_length, channels, ch_rot);
        
        dlib::matrix<std::complex<float>> lpf = dlib::create_fir_filter<std::complex<float>>(31, 4e6 / (float)sample_rate, dlib::blackman_nutall_window);
//        dlib::matrix<std::complex<float>> lpf = dlib::create_fft_fir_filter(ch_rot[0].nc(), 4e6 / (float)sample_rate, dlib::blackman_nutall_window);
        //dlib::matrix<std::complex<float>> lpf = dlib::create_fft_fir_filter(ch_rot[0].nc(), 4e6 / (float)sample_rate, dlib::blackman_nutall_window);

        double time_accum = 0.0;

        //dlib::matrix<std::complex<float>> x2(1, 1925);

        //for(kdx=0; kdx<100; ++kdx)
        //{
        //    start_time = chrono::high_resolution_clock::now();

        //        // create the random bit sequence
        //        for (idx = 0; idx < num_bits; ++idx)
        //            data[idx] = bits_gen(generator);


        //        dlib::matrix<std::complex<float>> iq_data = generate_oqpsk(data, amplitude, sample_rate, half_symbol_length);

        //        //dlib::matrix<std::complex<float>> temp = dlib::mat(iq_data);

        //        //dlib::fft_inplace(iq_data);

        //    for (jdx = 0; jdx < num_bursts; ++jdx)
        //    {

        //        ch_rnd = channel_gen(generator);



        //        //dlib::matrix<std::complex<double>> f_rot = dlib::generate_frequency_shift<double>(iq_data.size(), (double)channels[0], (double)sample_rate);


        //        iq_data = dlib::conv_same(iq_data, lpf);

        //        //iq_data = dlib::pointwise_multiply(iq_data, lpf);
        //        //dlib::ifft_inplace(iq_data);

        //        dlib::matrix<std::complex<float>> x3 = dlib::apply_frequency_shift(iq_data, ch_rot[ch_rnd]);
        //    }

        //    stop_time = chrono::high_resolution_clock::now();
        //    const auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
        //    std::cout << std::fixed << std::setprecision(16) << "time: " << int_ms.count()/1e6 << std::endl;

        //    time_accum += int_ms.count()/1e6;
        //}

        //std::cout << std::fixed << std::setprecision(16) << "aveage time: " << time_accum/100.0 << std::endl;

        //bp = 990;


        //dlib::matrix<double> hw = dlib::hann_window(N);


        //dlib::matrix<float> fir = dlib::create_fir_filter<float>(32, 0.75, dlib::hann_window);
        //dlib::matrix<std::complex<float>> fir_c = dlib::create_fir_filter<std::complex<float>>(32, 0.75, dlib::hann_window);


        //dlib::matrix<std::complex<float>> test2 = dlib::matrix_cast<std::complex<float>>(fir);


        //dlib::matrix<std::complex<float>> test3 = dlib::conv_same(fir, fir);
        //dlib::matrix<float> test3 = dlib::conv_same(fir, fir);

        //dlib::matrix<std::complex<float>> test3c = dlib::conv_same(test2, test2);

        //bool same_type = dlib::type_test(test2, test3);


        //std::cout << hw << std::endl;


        //std::cout << dlib::trans(fir_c) << std::endl;



        //dlib::matrix<std::complex<float>> test3c = apply_fir_filter(fir_c, 8, 0.5, dlib::hann_window);
        //std::cout << dlib::trans(test3c) << std::endl;


        //dlib::matrix<std::complex<float>> test4c = apply_frequency_shift(test3c, 1000, 10000);

        //auto res = dlib::apply_fir_filter(test3c, N, 0.5, dlib::hann_window<float>);

        //dlib::matrix<std::complex<float>> test4 = dlib::hann_window(N);
        
        //dlib::matrix<float> fir_c = dlib::create_fir_filter<std::complex<float>>(200, 0.75, dlib::hann_window<std::complex<float>>);


        //dlib::matrix<std::complex<float>> z_c = dlib::zeros_matrix<std::complex<float>>(N + 1, 1);

        bp = 4;

        cv::RNG rng2(1234567);

/*
        cv::VideoCapture cap(0, cv::CAP_ANY);  //Notebook camera input

        cv::VideoWriter writer;

        // Write this string to one line to be sure!!
        std::string writer_input = "appsrc ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! x264enc speed-preset=veryfast tune=zerolatency bitrate=800 ! rtspclientsink location=rtsp://localhost:8554/mystream ";
        writer.open(writer_input, 0, 30, cv::Size(640, 480), true);

        // Comment this line out
        cv::Mat cam_img;

        while(key != 'q')
        {
            if (!cap.isOpened())
            {
                std::cout << "Video Capture Fail" << std::endl;
                break;
            }
            cap.read(cam_img);

            cv::resize(cam_img, cam_img, cv::Size(640, 480));
            cv::imshow("raw", cam_img);
            writer << cam_img;
            key = cv::waitKey(25);
        }

        writer.release();
*/

        bp = 5;

        //console_input ci;

        //ci.arrow_key();

        //while (ci.get_running())
        //{

        //    if (ci.get_valid_input() == true)
        //    {
        //        k = (uint32_t)ci.get_input();
        //        //std::cout << "valid input: " << k << std::endl;

        //        if (k == 75)
        //        {
        //            position = std::max(0, --position);
        //        }
        //        else if (k == 77)
        //        {
        //            position = std::min(bar_width-1, ++position);
        //        }

        //        std::cout << "[" << std::string(0+position, '=') << "|" << std::string(bar_width - position - 1, '=') << "] " << 88.1 + 0.2*position << "  \r";
        //        std::cout.flush();


        //        
        //        ci.reset_valid_input();
        //    }



        //}

        bp = 3;

        //double a = 0.0;

        //start_time = chrono::high_resolution_clock::now();
        //for (jdx = 0; jdx < 0xFFFFFFFFFFFFFFFF; ++jdx)
        //{
        //    for (idx = 0; idx < 0xFFFFFFFFFFFFFFFF; ++idx)
        //    {
        //        a = std::exp(-0.6 * std::log(9.8));
        //    }
        //}
        //stop_time = chrono::high_resolution_clock::now();
        ////elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        //std::cout << std::fixed << std::setprecision(16) << "exp: " << a << " = " << elapsed_time.count() << std::endl;


        //start_time = chrono::high_resolution_clock::now();
        //for (jdx = 0; jdx < 0xFFFFFFFFFFFFFFFF; ++jdx)
        //{
        //    for (idx = 0; idx < 0xFFFFFFFFFFFFFFFF; ++idx)
        //    {
        //        a = std::pow(9.8, -0.6);
        //    }
        //}
        //stop_time = chrono::high_resolution_clock::now();
        ////elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        //std::cout << std::fixed << std::setprecision(16) << "pow: " << a << " = " << elapsed_time.count() << std::endl;

        //bp = 2;

        //std::cin.ignore();

        //cv::Mat img = cv::imread("D:/data/checker_board_512x512.png");

        //uint32_t kernel_size = 64;
        //double sigma = 5.0, weight = 0.6;
        //cv::Mat kernel, img1, img2, img3, img4;

        //create_gaussian_kernel(kernel_size, sigma, kernel);
        //img.convertTo(img, CV_32FC1);

        //cv::filter2D(img, img1, -1, kernel, cv::Point(-1, -1), 0.0, cv::BorderTypes::BORDER_REPLICATE);

        //sigma = 3.0;
        //create_gaussian_kernel(kernel_size, sigma, kernel);

        //cv::filter2D(img1, img2, -1, kernel, cv::Point(-1, -1), 0.0, cv::BorderTypes::BORDER_REPLICATE);

        //img3 = (weight / (2 * weight - 1)) * img1 - ((1 - weight) / (2 * weight - 1)) * img2;
        //img3.convertTo(img3, CV_8UC3);
        //
        //img4 = (1.0 + weight) * img1 - weight * img2;
        //img4.convertTo(img4, CV_8UC3);

        //bp = 1;

        //std::thread gi(get_input);

        //while (run)
        //{
        //    if (entry)
        //    {
        //        if (console_input == "test")
        //        {
        //            std::cout << "you pressed test" << std::endl;
        //        }
        //        else if (console_input == "stop")
        //        {
        //            std::cout << "stopping" << std::endl;
        //            run = false;
        //        }

        //        entry = false;
        //    }


        //}

        //gi.join();

        //cv::Mat cb;
        //generate_checkerboard(35, 35, 300, 150, cb);

        //cv::Mat X, Y;
        //meshgrid(cv::Range(0, 5), cv::Range(3, 6), X, Y);
        //
        //std::string mmap_name = "test";

        //std::vector<std::pair<uint8_t, uint8_t>> bg_br_table;

        //bg_br_table.push_back(std::make_pair(0, 1));
        //bg_br_table.push_back(std::make_pair(2, 3));
        //bg_br_table.push_back(std::make_pair(4, 5));

        //auto br_ptr = bg_br_table.data();

        //test_pair((void *)(br_ptr));
        //bp = 1;


        //mem_map mm(mmap_name, 256);

        //uint64_t position = 0;
        //double data = 21.112121212;
        //uint64_t t = 10000;
        //mm.write(position, data);
        //mm.write(position, t);
        //std::vector<float> v = { 0.21, 54656.25 };
        //mm.write_range(position, v);


        //mem_map mm2(mmap_name, 256);

        //uint64_t position2 = 0;
        //double data2;
        //uint64_t t2;
        //mm2.read(position2, data2);
        //mm2.read(position2, t2);
        //std::vector<float> v2;
        //mm.read_range(position2, 2, v2);


        //mm.close();
        //mm2.close();

        //std::cout << "done with test" << std::endl;
 /*       std::cout << std::string(argv[0]) << std::endl;

        test_inputfile = "../../rf_zsl/data/lfm_test_10M_100m_0000.bin";

        std::ifstream input_file;

        std::vector<int16_t> buffer;

        input_file.open(test_inputfile, std::ios::binary);
        if (!input_file.is_open())
            return 0;

        input_file.seekg(0, std::ios::end);
        size_t filesize = input_file.tellg();
        input_file.seekg(0, std::ios::beg);


        auto t4 = filesize / sizeof(int16_t) + (filesize % sizeof(int16_t) ? 1U : 0U);

        buffer.resize(filesize / sizeof(int16_t) + (filesize % sizeof(int16_t) ? 1U : 0U));

        input_file.read((char*)buffer.data(), filesize);

        bp = 2;*/

        uint32_t octaves = 3;
        double sn_scale = 0.020;
        double persistence = 50 / 100.0;
        std::vector<uint8_t> wood = { 41,44,35, 57,91,61, 80,114,113, 64,126,132 };

        // use these variables for the datatype > 0
        typedef void (*init_)(long seed);
        //typedef unsigned int (*evaluate_)(double x, double y, double scale, unsigned int num);
        //typedef unsigned int (*octave_evaluate_)(double x, double y, double scale, unsigned int octaves, double persistence);
        typedef void (*create_color_map_)(unsigned int h, unsigned int w, double scale, unsigned int octaves, double persistence, unsigned char* color, unsigned char* map);
        HINSTANCE simplex_noise_lib = NULL;
        init_ init;
        //evaluate_ evaluate;
        //octave_evaluate_ octave_evaluate;
        create_color_map_ create_color_map;

        std::string lib_filename = "D:/Projects/simplex_noise/build/Release/sn_lib.dll";

        simplex_noise_lib = LoadLibrary(lib_filename.c_str());

        if (simplex_noise_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        init = (init_)GetProcAddress(simplex_noise_lib, "init");
        create_color_map = (create_color_map_)GetProcAddress(simplex_noise_lib, "create_color_map");

        sn.init((long)time(NULL));

        octave_image = cv::Mat(600, 600, CV_8UC3);
        //create_color_map(img_h, img_w, sn_scale, octaves, persistence, wood.data(), random_img.data);

        cv::namedWindow(window_name, cv::WINDOW_NORMAL); // Create Window
        cv::resizeWindow(window_name, 1000, 800);

        std::string persistence_tb = "Persistence";
        std::string scale_tb = "Scale";
        std::string octave_tb = "Octave";

        cv::createTrackbar(persistence_tb, window_name, &per_slider, per_slider_max, on_trackbar);
        cv::createTrackbar(scale_tb, window_name, &scale_slider, scale_slider_max, on_trackbar);
        cv::createTrackbar(octave_tb, window_name, &octave_slider, octave_slider_max, on_trackbar);
        
        on_trackbar(0, 0);
        cv::waitKey(0);


        bp = 5;

        uint32_t intensity = (uint32_t)rnd.get_integer_in_range(2, 11);

        cv::RNG rng(1234567);

        //long nr = 600;
        //long nc = 800;
        //unsigned int N = 1200;
        //double scale = (double)60 / (double)nc;

        //generate_random_image(img, rng, nr, nc, N, scale);

        //bp = 0;

        //cv::Mat tf = cv::Mat(10, 10, CV_32FC1);

        //for (idx = 0; idx < 10; ++idx)
        //{
        //    for (jdx = 0; jdx < 10; ++jdx)
        //    {
        //        tf.at<float>(idx, jdx) = rng.uniform(0.0f, 50.0f);
        //    }
        //}

        // ----------------------------------------------------------------------------------------
        // test of target locator

        target_locator tl(1);

        observation o1(2.8 * 30, { 10 });
        observation o2(0.9 * 30, { 130 });

        tl.add_observation(o1);
        tl.add_observation(o2);

        tl.location.push_back(80.5);

        int32_t st = tl.get_position();

        bp = 1;
        
        // ----------------------------------------------------------------------------------------
        std::string test_file;
        std::string net_file;
        
        
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        net_file = "../../robot/common/nets/dc3_rgb_v10e_035_035_100_90_HPC_final_net.dat";
        
        // red backpack
        //test_file = "D:/Projects/object_detection_data/dc/train/full/backpack8-1.png";

        // blue backpack
        //test_file = "D:/Projects/object_detection_data/dc/train/full/backpack9-7.png";

        // black backpack
        // test_file = "D:/Projects/object_detection_data/dc/train/full/backpack7-1.png";
        // test_file = "D:/Projects/object_detection_data/dc/train/full/backpack2-1.png";

        // gray backpack
        // test_file = "D:/Projects/object_detection_data/dc/train/full/backpack1.png";
        //test_file = "D:/Projects/object_detection_data/dc/part4/image_0015.png";
        test_file = "D:/Projects/object_detection_data/dc/part5/image_0054.png";

#else
        net_file = "../../../dc_ws/src/robot/obj_det/nets/dc3_rgb_v10e_035_035_100_90_HPC_final_net.dat";
        
        test_file = "../../../dc_ws/image_0054.png";
        
#endif


        cv::Mat test_img = cv::imread(test_file);
        cv::Mat test_img2 = test_img.clone();
        cv::Rect roi;

        mouse_params mp;

        cv::namedWindow("My_Win", cv::WINDOW_NORMAL);

        cv::setMouseCallback("My_Win", mouse_click, (void*)&mp);

        // end selection process on SPACE (32) ESC (27) or ENTER (13)
        while (!(key == 32 || key == 27 || key == 13))
        {
            cv::rectangle(test_img2, mp.roi, cv::Scalar(255, 0, 0), 2, 1);

            cv::imshow("My_Win", test_img2);

            test_img2 = test_img.clone();

            key = cv::waitKey(10);
        }

        cv::destroyWindow("My_Win");


        bp = 9;

        //std::vector<float> gr_hist = { 0.00006, 0.00055, 0.00148, 0.00308, 0.00537, 0.00821, 0.01151, 0.01577, 0.01977, 0.02346, 0.02687, 0.03004, 0.03292, 0.03543, 0.03764, 0.03960, 0.04139, 0.04307, 0.04475, 0.04648, 0.04832, 0.05025, 0.05230, 0.05438, 0.05654, 0.05875, 0.06100, 0.06330, 0.06573, 0.06833, 0.07117, 0.07415, 0.07726, 0.08050, 0.08387, 0.08744, 0.09117, 0.09516, 0.09931, 0.10367, 0.10818, 0.11289, 0.11766, 0.12262, 0.12773, 0.13293, 0.13822, 0.14367, 0.14935, 0.15515, 0.16113, 0.16713, 0.17333, 0.17965, 0.18616, 0.19288, 0.19988, 0.20703, 0.21437, 0.22185, 0.22953, 0.23735, 0.24524, 0.25329, 0.26153, 0.26988, 0.27838, 0.28709, 0.29591, 0.30476, 0.31370, 0.32269, 0.33168, 0.34073, 0.34972, 0.35870, 0.36766, 0.37662, 0.38555, 0.39449, 0.40334, 0.41210, 0.42084, 0.42942, 0.43795, 0.44622, 0.45435, 0.46236, 0.47018, 0.47786, 0.48535, 0.49269, 0.49979, 0.50674, 0.51341, 0.51986, 0.52617, 0.53225, 0.53817, 0.54392, 0.54956, 0.55506, 0.56048, 0.56579, 0.57101, 0.57611, 0.58111, 0.58600, 0.59082, 0.59554, 0.60016, 0.60475, 0.60925, 0.61371, 0.61806, 0.62235, 0.62657, 0.63070, 0.63480, 0.63887, 0.64293, 0.64699, 0.65099, 0.65501, 0.65897, 0.66300, 0.66694, 0.67090, 0.67480, 0.67863, 0.68244, 0.68620, 0.68989, 0.69354, 0.69711, 0.70068, 0.70412, 0.70756, 0.71098, 0.71421, 0.71745, 0.72058, 0.72363, 0.72675, 0.72969, 0.73268, 0.73559, 0.73850, 0.74148, 0.74449, 0.74756, 0.75056, 0.75346, 0.75633, 0.75908, 0.76179, 0.76442, 0.76702, 0.76962, 0.77215, 0.77462, 0.77713, 0.77961, 0.78212, 0.78458, 0.78708, 0.78963, 0.79216, 0.79479, 0.79752, 0.80036, 0.80324, 0.80609, 0.80904, 0.81200, 0.81499, 0.81800, 0.82096, 0.82395, 0.82681, 0.82958, 0.83227, 0.83479, 0.83714, 0.83934, 0.84135, 0.84326, 0.84499, 0.84666, 0.84820, 0.84967, 0.85108, 0.85243, 0.85379, 0.85518, 0.85653, 0.85788, 0.85921, 0.86051, 0.86176, 0.86292, 0.86401, 0.86506, 0.86608, 0.86706, 0.86803, 0.86898, 0.86987, 0.87074, 0.87158, 0.87243, 0.87330, 0.87416, 0.87502, 0.87585, 0.87662, 0.87735, 0.87806, 0.87876, 0.87947, 0.88019, 0.88094, 0.88170, 0.88248, 0.88332, 0.88421, 0.88519, 0.88623, 0.88730, 0.88842, 0.88955, 0.89068, 0.89185, 0.89300, 0.89407, 0.89511, 0.89608, 0.89701, 0.89789, 0.89870, 0.89951, 0.90032, 0.90114, 0.90197, 0.90285, 0.90389, 0.90508, 0.90633, 0.90776, 0.90942, 0.91150, 0.91410, 0.91681, 0.92039, 0.92648, 1.00000 };
        dlib::matrix<uint8_t> gr_hist = { 0, 0, 0, 0, 1, 2, 2, 4, 5, 5, 6, 7, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 49, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 75, 77, 79, 82, 84, 86, 89, 91, 93, 96, 98, 100, 102, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 130, 132, 134, 135, 137, 138, 140, 141, 142, 144, 145, 146, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 174, 175, 176, 177, 178, 179, 180, 181, 182, 182, 183, 184, 185, 186, 186, 187, 188, 189, 189, 190, 191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 198, 198, 199, 200, 200, 201, 202, 202, 203, 204, 204, 205, 206, 207, 207, 208, 209, 210, 210, 211, 212, 212, 213, 214, 214, 215, 215, 215, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 220, 221, 221, 221, 221, 222, 222, 222, 222, 222, 223, 223, 223, 223, 223, 224, 224, 224, 224, 224, 225, 225, 225, 225, 225, 226, 226, 226, 227, 227, 227, 227, 228, 228, 228, 228, 229, 229, 229, 229, 230, 230, 230, 230, 231, 231, 231, 232, 233, 233, 234, 236, 254 };


        int crop_x = 0;
        int crop_y = 0;
        int crop_w = 1280;
        int crop_h = 720;
        dlib::rectangle crop_rect(crop_x, crop_y, crop_x + crop_w - 1, crop_y + crop_h - 1);

        std::vector<cv::Scalar> class_color;
        class_color.push_back(cv::Scalar(0, 255, 0));
        class_color.push_back(cv::Scalar(0, 0, 255));

/*
        anet_type test_net;
        dlib::deserialize(net_file) >> test_net;

        // get the details about the loss layer -> the number and names of the classes
        dlib::mmod_options options = dlib::layer<0>(test_net).loss_details().get_options();

        std::set<std::string> tmp_names;
        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        for (uint64_t idx = 0; idx < options.detector_windows.size(); ++idx)
        {
            std::cout << "detector window (w x h): " << options.detector_windows[idx].label << " - " << options.detector_windows[idx].width
                << " x " << options.detector_windows[idx].height << std::endl;
            tmp_names.insert(options.detector_windows[idx].label);
        }
        std::cout << "------------------------------------------------------------------" << std::endl;

        // pull out the class names
        std::vector<std::string> class_names;
        for (const auto& it : tmp_names)
        {
            class_names.push_back(it);
        }

*/
        cv::Mat cv_img = cv::imread(test_file, cv::IMREAD_COLOR);
        //cv::Mat cv_img = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(255, 255, 255));
        
        cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
        
        dlib::matrix<dlib::rgb_pixel> rgb_img, rgb_img2;

        dlib::assign_image(rgb_img, dlib::cv_image<dlib::rgb_pixel>(cv_img));

        rgb_img = dlib::subm(rgb_img, crop_rect);

        histogram_specification(rgb_img, rgb_img2, gr_hist);


        // ----------------------------------------------------------------------------------------


        bp = 2;
        // ----------------------------------------------------------------------------------------
        //dlib::matrix<dlib::rgb_pixel> color_map;
        //dlib::matrix<uint8_t> map, map2;


        //dlib::load_image(color_map, "../test_map.png");

        //dlib::assign_image(map, color_map);

        //dlib::matrix<uint32_t> input(1, 28);
        //input = 11, 10, 9, 8, 8, 8, 8, 9, 10, 11, 14, 18, 29, 80, 75, 26, 16, 12, 10, 8, 8, 7, 7, 7, 7, 8, 8, 10;
        //dlib::matrix<uint32_t> input(1, 7);
        //input = 11, 8, 11, 80, 10, 7, 10;
        dlib::matrix<float, 1, 10> input = dlib::ones_matrix<float>(1, 10);

        dlib::matrix<float> motion(2, 1);
        motion = 1.0, 1.0;

        //car_net_type c_net;

        std::cout << c_net << std::endl;

        double intial_learning_rate = 0.0001;
        dlib::dnn_trainer<car_net_type, dlib::adam> trainer(c_net, dlib::adam(0.0001, 0.9, 0.99), { 0 });
        trainer.set_learning_rate(intial_learning_rate);
        trainer.be_verbose();
        //trainer.set_synchronization_file((sync_save_location + net_sync_name), std::chrono::minutes(5));
        //trainer.set_iterations_without_progress_threshold(tp.steps_wo_progess);
        trainer.set_test_iterations_without_progress_threshold(5000);
        //trainer.set_learning_rate_shrink_factor(tp.learning_rate_shrink_factor);

        std::cout << trainer << std::endl;

        auto t1a = dlib::layer<2>(c_net).layer_details().get_weights();
        auto &t1a1 = dlib::layer<2>(c_net);

        dlib::layer<2>(c_net).layer_details().setup(t1a1.subnet());
        auto t1b = dlib::layer<2>(c_net).layer_details().get_weights();

        for (idx = 0; idx < 10; ++idx)
        {
            trainer.train_one_step({ input }, { motion });
        }
        //auto &t1c = dlib::layer<2>(c_net).layer_details().get_weights();
        //trainer.train_one_step({ input }, { motion });


        dlib::net_to_xml(c_net, "car_net.xml");

        //auto c_net2 = c_net.subnet();

        //std::cout << std::endl << c_net2 << std::endl;


        //auto &t2a = dlib::layer<2>(c_net).layer_details();
        //auto t2b= t2a.get_layer_params();

        //uint64_t l2_size = dlib::layer<2>(c_net).layer_details().get_layer_params().size();
        //auto l2_data = dlib::layer<2>(c_net).layer_details().get_layer_params().host();

        // this is how to copy values into the network
        //uint32_t index=0;
        //for (idx = 0; idx < l2_size; ++idx,++index)
        //    *(l2_data + idx) = index;

        //dlib::net_to_xml(c_net, "car_net2.xml");

        //auto& t5a = dlib::layer<3>(c_net).layer_details();
        //auto& t5b = t5a.get_layer_params();
        //auto t5c = t5b.host();


        
        // ----------------------------------------------------------------------------------------

        //start_time = chrono::system_clock::now(); 
        //auto result = dlib::find_min_global(schwefel2,
        //    {-500, -500, -500, -500}, // lower bounds
        //    { 500, 500, 500, 500 }, // upper bounds
        //    dlib::max_function_calls(2000));
        //stop_time = chrono::system_clock::now();
        //elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        //cout.precision(6);
        //std::cout << "find_min_global (" << elapsed_time.count() << "): " << result.x << ", " << result.y << std::endl;

        // ----------------------------------------------------------------------------------------
        //dlib::load_image(color_map, "../test_map_v2_2.png");

        //dlib::pso_options psooptions(100, 5000, 2.4, 2.1, 1.0, 1, 1.0);

        //std::cout << "----------------------------------------------------------------------------------------" << std::endl;
        //std::cout << psooptions << std::endl;

        // schwefel(dlib::matrix<double> x)

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

