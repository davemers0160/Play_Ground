#define _CRT_SECURE_NO_WARNINGS

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

// OpenCV includes
#include <opencv2/core.hpp>           
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp> 
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>

// custom includes
//#include "mmaplib.h"
#include "pg.h"


#include "../../robot/obj_det/include/obj_det_net_rgb_v10.h"


//#include "mmap.h"
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "read_binary_image.h" 
//#include "make_dir.h"
#include "file_ops.h"
#include "ssim.h"
#include "dlib_matrix_threshold.h"
#include "gorgon_capture.h"
#include "modulo.h"
#include "overlay_bounding_box.h"
#include "simplex_noise.h"
#include "ocv_threshold_functions.h"

#include "pso.h"
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

// Network includes
//#include "dfd_net_v14.h"
//#include "dfd_net_v14_pso_01.h"
//#include "dfd_net_rw_v19.h"
//#include "load_dfd_rw_data.h"
//#include "load_dfd_data.h"
//#include "resnet101_v1.h"

//#include "cyclic_analysis.h"

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

const int scale_slider_max = 50;
int scale_slider = 8;

const int octave_slider_max = 30;
int octave_slider = 5;

const int per_slider_max = 100;
int per_slider = 6;

std::vector<cv::Vec3b> color = { cv::Vec3b(41,44,35), cv::Vec3b(57,91,61), cv::Vec3b(80,114,113), cv::Vec3b(64,126,132) };

open_simplex_noise sn;


static void on_trackbar(int, void*)
{
    double v;
    uint32_t r, c;

    double scale = 1.0/(double)(scale_slider+1);

    double p = (double)(per_slider + 1) / 100.0;

    for (r = 0; r < octave_image.rows; ++r)
    {
        for (c = 0; c < octave_image.cols; ++c)
        {
            v = sn.octave((double)r * scale, (double)c * scale, octave_slider+1, p*1.0);
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

// ----------------------------------------------------------------------------------------
void generate_random_image(
    cv::Mat& img,
    cv::RNG rng,
    long nr, 
    long nc, 
    unsigned int N, 
    double scale   
)
{
    unsigned int idx;

    // get the random background color
    cv::Scalar bg_color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

    // create the image with the random background color
    img = cv::Mat(nr, nc, CV_8UC3, bg_color);

    // create N shapes
    for (idx = 0; idx < N; ++idx)
    {

        // get the random color for the shape
        cv::Scalar C = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // make sure the color picked is not the background color
        while (C == bg_color)
        {
            C = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        }

        // generate the random point
        long x = rng.uniform(0, nc);
        long y = rng.uniform(0, nr);
        long r1, r2, h, w, s;
        double a;

        cv::RotatedRect rect;
        cv::Point2f vertices2f[4];
        cv::Point vertices[4];
        //cv::Point pt[2][3];
        std::vector<cv::Point> pts(3);
        std::vector<std::vector<cv::Point> > vpts(1);
        vpts.push_back(pts);


        long x_1; //= -window_width / 2;
        long x_2; //= window_width * 3 / 2;
        long y_1; //= -window_width / 2;
        long y_2; //= window_width * 3 / 2;

        // get the shape type
        switch (rng.uniform(0, 3))
        {
        case 0:
            
            // pick a random radi for the ellipse
            r1 = std::floor(0.5 * scale * rng.uniform(0, std::min(nr, nc)));
            r2 = std::floor(0.5 * scale * rng.uniform(0, std::min(nr, nc)));
            a = rng.uniform(0.0, 360.0);

            cv::ellipse(img, cv::Point(x, y), cv::Size(r1, r2), a, 0.0, 360.0, C, -1, cv::LineTypes::LINE_8, 0);
            break;

        case 1:

            h = std::floor(scale * rng.uniform(0, std::min(nr, nc)));
            w = std::floor(scale * rng.uniform(0, std::min(nr, nc)));
            a = rng.uniform(0.0, 360.0);

            // Create the rotated rectangle
            rect = cv::RotatedRect(cv::Point(x,y), cv::Size(w,h), a);

            // We take the edges that OpenCV calculated for us
            rect.points(vertices2f);

            // Convert them so we can use them in a fillConvexPoly
            for (int jdx = 0; jdx < 4; ++jdx) 
            {
                vertices[jdx] = vertices2f[jdx];
            }

            // Now we can fill the rotated rectangle with our specified color
            cv::fillConvexPoly(img, vertices, 4, C);
            break;

        case 2:

            s = rng.uniform(3, 9);
            x_1 = -(0.5 * scale * nc);
            x_2 = (1.5 * scale * nc);
            y_1 = -(0.5 * scale * nr);
            y_2 = (1.5 * scale * nc);

            pts.clear();

            for (int jdx = 0; jdx < s; ++jdx)
            {
                pts.push_back(cv::Point(rng.uniform(x_1, x_2) + x, rng.uniform(y_1, y_2) + y));
            }

            vpts[0] = pts;
            cv::fillPoly(img, vpts, C, cv::LineTypes::LINE_8, 0);

            break;

        }

    }

}


void generate_random_image(unsigned char*& img,
    long long seed,
    long nr,
    long nc,
    unsigned int N,
    double scale
)
{
    cv::RNG rng(seed);

    cv::Mat cv_img;

    generate_random_image(cv_img, rng, nr, nc, N, scale);

    //    memcpy((void*)data_params, &network_output(0, 0), network_output.size() * sizeof(float));
    img = new unsigned char[cv_img.total()*3];
    memcpy((void*)img, cv_img.ptr<unsigned char>(0), cv_img.total()*3);

    int bp = 0;
}


bool operator<(const dlib::hsi_pixel& p1, const dlib::hsi_pixel& p2)
{
    if ((p1.h < p2.h) && (p1.s < p2.s) && (p1.i < p2.i))
        return true;

    return false;
}

bool operator>(const dlib::hsi_pixel& p1, const dlib::hsi_pixel& p2)
{
    if ((p1.h > p2.h) && (p1.s > p2.s) && (p1.i > p2.i))
        return true;

    return false;
}

bool operator<=(const dlib::hsi_pixel& p1, const dlib::hsi_pixel& p2)
{
    if ((p1.h <= p2.h) && (p1.s <= p2.s) && (p1.i <= p2.i))
        return true;

    return false;
}

bool operator>=(const dlib::hsi_pixel& p1, const dlib::hsi_pixel& p2)
{
    if ((p1.h >= p2.h) && (p1.s >= p2.s) && (p1.i >= p2.i))
        return true;

    return false;
}

dlib::matrix<uint32_t, 1, 4> get_color_match(dlib::matrix<dlib::rgb_pixel>& img, dlib::mmod_rect& det)
{
    uint64_t r, c;

    dlib::hsi_pixel red_ll1(0, 0, 64);
    dlib::hsi_pixel red_ul1(15, 255, 255);
    dlib::hsi_pixel red_ll2(240, 0, 64);
    dlib::hsi_pixel red_ul2(255, 255, 255);

    dlib::hsi_pixel blue_ll(155, 0, 64);
    dlib::hsi_pixel blue_ul(185, 255, 255);

    dlib::hsi_pixel black_ll(0, 48, 0);
    dlib::hsi_pixel black_ul(255, 128, 64);

    dlib::hsi_pixel gray_ll(0, 0, 40);
    dlib::hsi_pixel gray_ul(255, 255, 128);

    const int w = 20, h = 20;

    dlib::matrix<uint8_t> red_mask = dlib::zeros_matrix<uint8_t>(h, w);
    dlib::matrix<uint8_t> blue_mask = dlib::zeros_matrix<uint8_t>(h, w);
    dlib::matrix<uint8_t> black_mask = dlib::zeros_matrix<uint8_t>(h, w);
    dlib::matrix<uint8_t> gray_mask = dlib::zeros_matrix<uint8_t>(h, w);

    // crop out the detection
    dlib::point ctr = dlib::center(det.rect);

    dlib::matrix<dlib::rgb_pixel> obj_crop = dlib::subm(img, dlib::centered_rect(ctr, w, h));
    dlib::matrix<dlib::hsi_pixel> hsi_crop;
    dlib::assign_image(hsi_crop, obj_crop);

    dlib::hsi_pixel p;

    for (int r = 0; r < hsi_crop.nr(); ++r)
    {
        for (int c = 0; c < hsi_crop.nc(); ++c)
        {
            dlib::assign_pixel(p, hsi_crop(r, c));

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
            else if ((p >= black_ll) && (p <= black_ul))
            {
                black_mask(r, c) = 1;
            }
            else if ((p >= gray_ll) && (p <= gray_ul))
            {
                gray_mask(r, c) = 1;
            }

        }
    }

    //double red_sum = dlib::sum(red_mask);
    //double blue_sum = dlib::sum(blue_mask);
    //double black_sum = dlib::sum(black_mask);
    //double gray_sum = dlib::sum(gray_mask);

    dlib::matrix<uint32_t, 1, 4> res;
    res = (uint32_t)dlib::sum(red_mask), (uint32_t)dlib::sum(blue_mask), (uint32_t)dlib::sum(black_mask), (uint32_t)dlib::sum(gray_mask);

    return res;

}




// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx=0, jdx=0;
    uint64_t r, c;

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
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

#else
        std::cout << "argv[0]: " << std::string(argv[0]) << std::endl;
        std::string exe_path = get_linux_path();
        std::cout << "Path: " << exe_path << std::endl;
#endif

        int bp = 0;

        uint32_t intensity = (uint32_t)rnd.get_integer_in_range(2, 11);

        cv::Mat img;

        cv::RNG rng(123456);

        long nr = 500;
        long nc = 500;
        unsigned int N = 800;
        double scale = (double)50 / (double)nc;

        generate_random_image(img, rng, nr, nc, N, scale);

        bp = 0;

        cv::Mat tf = cv::Mat(10, 10, CV_32FC1);

        for (idx = 0; idx < 10; ++idx)
        {
            for (jdx = 0; jdx < 10; ++jdx)
            {
                tf.at<float>(idx, jdx) = rng.uniform(0.0f, 50.0f);
            }
        }


        cv::Scalar mn = cv::mean(tf);

        std::cout << "mean: " << mn[0] << std::endl;
        double mn2 = nan_mean<float>(tf);

        tf.at<float>(5, 5) = nan("");

        mn2 = nan_mean<float>(tf);


        cv::patchNaNs(tf);
        cv::Scalar mn3 = cv::mean(tf);


        cv::Mat tf4;
        float min_val = 0.0;
        float max_val = 25.0;

        ranged_threshold<float>(tf, tf4, 0.0, 25.0);


        unsigned char* test;

        //generate_random_image(test, 123456, nr, nc, N, scale);

        bp = 1;

        std::string net_file = "../../robot/obj_det/nets/dc3_rgb_v10e_035_035_100_90_HPC_final_net.dat";

        // red backpack
        std::string test_file = "D:/Projects/object_detection_data/dc/train/full/backpack8-1.png";

        // blue backpack
        //std::string test_file = "D:/Projects/object_detection_data/dc/train/full/backpack9-7.png";

        // black backpack
        //std::string test_file = "D:/Projects/object_detection_data/dc/train/full/backpack7-1.png";

        // gray backpack
        //std::string test_file = "D:/Projects/object_detection_data/dc/train/full/backpack1.png";
        //std::string test_file = "D:/Projects/object_detection_data/dc/part4/image_0015.png";

        int crop_x = 270;
        int crop_y = 0;
        int crop_w = 720;
        int crop_h = 720;
        dlib::rectangle crop_rect(crop_x, crop_y, crop_x + crop_w - 1, crop_y + crop_h - 1);

        std::vector<cv::Scalar> class_color;
        class_color.push_back(cv::Scalar(0, 255, 0));
        class_color.push_back(cv::Scalar(0, 0, 255));


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


        cv::Mat cv_img = cv::imread(test_file, cv::IMREAD_COLOR);
        //cv::Mat cv_img = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(255, 255, 255));
        
        cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
        
        dlib::matrix<dlib::rgb_pixel> rgb_img;

        dlib::assign_image(rgb_img, dlib::cv_image<dlib::rgb_pixel>(cv_img));

        rgb_img = dlib::subm(rgb_img, crop_rect);

        std::vector<dlib::mmod_rect> d = test_net(rgb_img);

        //d.push_back(dlib::mmod_rect(dlib::rectangle(30, 30, 60, 60), 1.0, "backpack"));

        for (idx = 0; idx < d.size(); ++idx)
        {
            auto class_index = std::find(class_names.begin(), class_names.end(), d[idx].label);

            if (d[idx].label == "backpack")
            {
                dlib::matrix<uint32_t, 1, 4> cm = get_color_match(rgb_img, d[idx]);


                long index = dlib::index_of_max(cm);

                if (index == 3)
                {
                    std::cout << "gray backpack" << std::endl;
                }

                bp = 5;
            }

            d[idx].rect = dlib::translate_rect(d[idx].rect, crop_x, crop_y);
            overlay_bounding_box(cv_img, dlib2cv_rect(d[idx].rect), d[idx].label, class_color[std::distance(class_names.begin(), class_index)]);

        }

        //dlib::image_window win;

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

        auto &t1a = dlib::layer<2>(c_net).layer_details().get_weights();
        auto &t1a1 = dlib::layer<2>(c_net);

        dlib::layer<2>(c_net).layer_details().setup(t1a1.subnet());
        auto &t1b = dlib::layer<2>(c_net).layer_details().get_weights();

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
        dlib::load_image(color_map, "../test_map_v2_2.png");

        dlib::pso_options psooptions(100, 5000, 2.4, 2.1, 1.0, 1, 1.0);

        std::cout << "----------------------------------------------------------------------------------------" << std::endl;
        std::cout << psooptions << std::endl;

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

