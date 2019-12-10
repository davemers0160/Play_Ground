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
#include <array>
#include <algorithm>
#include <type_traits>

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
#include "overlay_bounding_box.h"

#include "pso.h"
#include "pso_particle.h"
//#include "ycrcb_pixel.h"
//#include "dfd_array_cropper.h"
#include "rot_90.h"
//#include "dlib_srelu.h"
//#include "dlib_elu.h"
#include "center_cropper.h"
//#include "dfd_cropper.h"

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

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
extern const uint32_t array_depth;
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
const char* wndname = "Square Detection Demo";

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

// ----------------------------------------------------------------------------------------

using car_net_type = dlib::loss_mean_squared_multioutput<dlib::htan<dlib::fc<2,
    dlib::htan<dlib::fc<10,
    dlib::fc<50,
    dlib::fc<250,
    dlib::input<dlib::matrix<float>>
    > > >> >>>;

car_net_type c_net;
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

// ----------------------------------------------------------------------------------------

class vehicle
{

public:
	
	uint8_t threshold = 240;
	double heading;
    const unsigned long width = 6;
    const unsigned long length = 8;
    const uint32_t radius = 1;

    const double L = 6;     // wheel base
    const double r = 1;      // wheel radius

    double points;

    dlib::point C;

	float max_range = 80.0;
    //std::vector<double> detection_angles = { -135, -90, -45, 0, 45, 90, 135 };
    //std::vector<double> detection_angles = {-135, -125, -115, -105, -95, -85, -75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135};
    std::vector<double> detection_angles = { -90.0, -45.0, -22.5, -11.0, -0.5, 0.5, 11.0, 22.5, 45.0, 90.0 };

    //std::vector<double> detection_angles;

	std::vector<float> detection_ranges;

    vehicle(dlib::point C_, double heading_)
    {
        C = C_;

        heading = heading_*(dlib::pi / 180.0);

        points = 0.0;

        //for (double idx = -90; idx <= 90; idx += 22.5)
        for (uint32_t idx = 0; idx < detection_angles.size(); ++idx)
        {
            detection_angles[idx] = detection_angles[idx] * (dlib::pi / 180.0);
            //detection_angles.push_back(idx * (dlib::pi / 180.0));
        }

        detection_ranges.resize(detection_angles.size());
    }

	void get_ranges(dlib::matrix<uint8_t> map, dlib::matrix<uint8_t> &map2)
	{
		uint32_t idx;
		uint32_t x, y;

        uint32_t map_width = map.nc();
        uint32_t map_height = map.nr();

        //double h = heading + dlib::pi;

        dlib::assign_image(map2, map);

        map2(C.y(), C.x()) = 200;
		
		for(idx=0; idx< detection_angles.size(); ++idx)
		{
			detection_ranges[idx] = 1.0;
			
			for(uint32_t r=1; r<max_range; ++r)
			{

                x = (uint32_t)floor(r*std::cos(heading + detection_angles[idx]) + C.x() + 0.5);
                y = (uint32_t)floor(r*std::sin(heading + detection_angles[idx]) + C.y() + 0.5);

				
                if (x<0 || x>(map_width-1))
                {
                    detection_ranges[idx] = r/ max_range;
                    break;
                }

                if (y<0 || y>(map_height-1))
                {
                    detection_ranges[idx] = r/ max_range;
                    break;
                }

				if(map(y,x) > threshold)
				{
					detection_ranges[idx] = r/ max_range;
                    break;
				}
                map2(y, x) = 128;

			}
			
		}
	}   // end of get_ranges
	
    // ----------------------------------------------------------------------------------------

    //void move(double vl, double vr)
    //{
    //    double w = (vr - vl) / L;
    //    double h = heading + dlib::pi;
    //    double R = 0.0;
    //    
    //    long x_p = 0;
    //    long y_p = 0;

    //    if (std::abs(vr - vl) < 0.01)
    //    {
    //        x_p = C.x() + r * vr * std::cos(h);
    //        y_p = C.y() + r * vr * std::sin(h);
    //    }
    //    else
    //    {
    //        R = (L / 2.0) * (vr + vl);

    //        double ICCx = C.x() - R * std::sin(h);
    //        double ICCy = C.y() + R * std::cos(h);

    //        x_p = R * std::sin(h) * std::cos(w) + R * std::cos(h) * std::sin(w) + ICCx;
    //        y_p = R * std::sin(h) * std::sin(w) - R * std::cos(h) * std::cos(w) + ICCy;
    //    }

    //    C = dlib::point(x_p, y_p);

    //    heading += w;

    //}   // end of move


    void move(double fb, double lr)
    {

        //heading += lr * (dlib::pi * 0.0027777777778);    // 1/360
        heading += lr * (dlib::pi * 0.0055555555556);    // 2/360 
        //heading += lr * (dlib::pi * 0.0083333333333);    // 3/360 
        //heading += lr * (dlib::pi * 0.0138888888889);    // 5/360

        if (heading >= 2.0*dlib::pi)
            heading -= 2.0*dlib::pi;
        else if(heading <= -2.0*dlib::pi)
            heading += 2.0*dlib::pi;

        long x_p = C.x() + std::floor(fb * std::cos(heading) + 0.5);
        long y_p = C.y() + std::floor(fb * std::sin(heading) + 0.5);

        C = dlib::point(x_p, y_p);

    }   // end of move

    // ----------------------------------------------------------------------------------------
    
    bool test_for_crash(dlib::matrix<uint8_t> map)
    {
        bool crash = false;

        //double h = heading - dlib::pi / 2.0;
        double h = heading;

        long Lx = C.x() - floor((L / 2.0) * std::cos(h) + 0.5);
        long Ly = C.y() - floor((L / 2.0) * std::sin(h) + 0.5);

        long Rx = C.x() + floor((L / 2.0) * std::cos(h) + 0.5);
        long Ry = C.y() + floor((L / 2.0) * std::sin(h) + 0.5);

        //map(C.y(), C.x()) = 200;

        if ((C.x() < 1) || C.x() >= map.nc() - 1)
            crash = true;

        else if ((C.y() < 1) || C.y() >= map.nr() - 1)
            crash = true;

        else if ((Lx < 1) || Lx >= map.nc()-1)
            crash = true;

        else if ((Ly < 1) || Ly >= map.nr()-1)
            crash = true;

        else if ((Rx < 1) || Rx >= map.nc()-1)
            crash = true;

        else if ((Ry < 1) || Ry >= map.nr()-1)
            crash = true;

        else if (map(Ly, Lx) > threshold)
        {
            crash = true;
            map(Ly, Lx) = 100;
            map(Ry, Rx) = 150;
        }
        else if (map(Ry, Rx) > threshold)
        {
            crash = true;
            map(Ly, Lx) = 100;
            map(Ry, Rx) = 150;
        }
        else
        {
            map(Ly, Lx) = 100;
            map(Ry, Rx) = 150;
        }

        return crash;

    }   // end of test_for_crash

    void check_for_points(dlib::matrix<uint8_t> &map)
    {
        if (map(C.y(), C.x()) == 85)
        {
            points += 10;
            clear_line(map);
        }
    }


private:

    void clear_line(dlib::matrix<uint8_t> &map)
    {

        long offset = 4;

        //long x = C.x() - offset; // 11
        //long y = C.y() - offset; // 11

        long x1 = std::min(std::max(C.x() - offset, 0L), map.nc() - 1);
        long y1 = std::min(std::max(C.y() - offset, 0L), map.nr() - 1);

        long x2 = std::min(C.x() + offset, map.nc() - 1); // 22
        long y2 = std::min(C.y() + offset, map.nr() - 1); // 22

        dlib::rectangle rect(x1, y1, x2, y2);

        dlib::matrix<uint8_t> sm = dlib::subm(map, rect);
        threshold_to_zero(sm, sm, 90, false);

        dlib::set_subm(map, rect) = sm;

    }   // end of get_points
};

// ----------------------------------------------------------------------------------------

double eval_net(particle p)
{
    long idx;
    dlib::matrix<uint8_t> map, map2;

    dlib::point starting_point = dlib::point(12, 10);
    uint64_t current_points = 0;
    uint64_t moves_without_points = 0;

    dlib::assign_image(map, color_map);

    vehicle vh1(starting_point, 90.0);

    bool crash = false;

    long l2_size = dlib::layer<2>(c_net).layer_details().get_layer_params().size();
    auto l2_data = dlib::layer<2>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x1 = p.get_x1();

    // copy values into the network
    for (idx = 0; idx < l2_size; ++idx)
        *(l2_data + idx) = (float)x1(0,idx);

    long l3_size = dlib::layer<4>(c_net).layer_details().get_layer_params().size();
    auto l3_data = dlib::layer<4>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x2 = p.get_x2();

    for (idx = 0; idx < l3_size; ++idx)
        *(l3_data + idx) = (float)x2(0, idx);

    long l4_size = dlib::layer<5>(c_net).layer_details().get_layer_params().size();
    auto l4_data = dlib::layer<5>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x3 = p.get_x3();

    for (idx = 0; idx < l4_size; ++idx)
        *(l4_data + idx) = (float)x3(0, idx);

    long l5_size = dlib::layer<6>(c_net).layer_details().get_layer_params().size();
    auto l5_data = dlib::layer<6>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x4 = p.get_x4();

    for (idx = 0; idx < l5_size; ++idx)
        *(l5_data + idx) = (float)x4(0, idx);

    uint64_t movement_count = 0;
    
    while (crash == false)
    {
        //current_points = vh1.points;
        
        vh1.check_for_points(map);
        vh1.get_ranges(map, map2);
        win.clear_overlay();
        win.set_image(map2);

        dlib::matrix<float> m3 = dlib::trans(dlib::mat(vh1.detection_ranges));

        //std::cout << dlib::csv << m3 << std::endl;

        dlib::matrix<float> m2 = c_net(m3);

        vh1.move(m2(0, 0), m2(1, 0));

        //vh1.move(2 * 1, 0);
        //vh1.move(2 * 1, -0.5);

        //vh1.move(2 * -1, 0.5);
        //vh1.move(2 * -1, -0.5);

        std::string title = "Particle Number: " + num2str(p.get_number(), "%03d") + ", B: " + num2str(vh1.heading*180.0/dlib::pi, "%2.4f") + ", L/R: " + num2str(m2(0, 0), "%2.4f/") + num2str(m2(1, 0), "%2.4f") + ", Points: " + num2str(-vh1.points, "%4.0f");
        win.set_title(title);

        if(current_points == vh1.points)
        {
            ++movement_count;
        }
        else
        {
            current_points = vh1.points;
            movement_count = 0;
        }

        crash = vh1.test_for_crash(map);

        if (movement_count > 1000)
        {
            std::cout << "Count" << std::endl;
            crash = true;
        }

    }

    std::cout << "Particle Number: " << num2str(p.get_number(), "%03d") << ", Points: " << -vh1.points << std::endl;
    //dlib::sleep(200);
    return -vh1.points;
}

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
            std::cout << "argv[0]: " << std::string(argv[0]) << std::endl;
            std::string exe_path = get_linux_path();
            std::cout << "Path: " << exe_path << std::endl;
        #endif  

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

        //vehicle vh1(dlib::point(11,10), 270);
        //bool crash = false;

        //vh1.check_for_points(map);
        //crash = vh1.test_for_crash(map);

        //vh1.get_ranges(map);
        //vh1.move(2, 2);

        //crash = vh1.test_for_crash(map);
        //vh1.get_ranges(map);
        //vh1.move(2, -2);

        //crash = vh1.test_for_crash(map);
        //vh1.get_ranges(map);
        //vh1.move(2, 2);

        //crash = vh1.test_for_crash(map);
        //vh1.get_ranges(map);
        //vh1.move(-2, 2);

        //crash = vh1.test_for_crash(map);
        //vh1.get_ranges(map);
        //vh1.move(-2, -2);
        //dlib::image_window win;
        //while (crash == false)
        //{
        //    vh1.check_for_points(map);
        //    vh1.get_ranges(map, map2);
        //    win.clear_overlay();
        //    win.set_image(map2);

        //    dlib::matrix<uint32_t> m3 = dlib::trans(dlib::mat(vh1.detection_ranges));
        //    
        //    std::cout << dlib::csv << m3 << std::endl;

        //    dlib::matrix<float> m2 = c_net(m3);

        //    vh1.move(2*m2(0, 0), 2*m2(1, 0));

        //    crash = vh1.test_for_crash(map);

        //    dlib::sleep(750);
        //
        //}

        //particle ptcl;
        //double result = eval_net(ptcl);

        //vh1.move(-1.0, 2.0);

        //crash = vh1.test_for_crash(map);
        //vh1.get_ranges(map,map2);

        //vh1.move(12, 12);
        //crash = vh1.test_for_crash(map);
        //vh1.get_ranges(map,map2);

        
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

        dlib::pso_options options(40, 400, 2.4, 2.1, 1.0, 1, 1.0);

        std::cout << "----------------------------------------------------------------------------------------" << std::endl;
        std::cout << options << std::endl;

        // schwefel(dlib::matrix<double> x)

        dlib::pso<particle> p(options);
        //p.set_syncfile("test.dat");

        //dlib::matrix<double, 1, 2> x1,x2, v1,v2;
        dlib::matrix<double, 1, fc1_size> x1,v1;
        dlib::matrix<double, 1, fc2_size> x2,v2;
        dlib::matrix<double, 1, fc3_size> x3,v3;
        dlib::matrix<double, 1, fc4_size> x4,v4;

        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = 5.0;
            v1(0, idx) = 0.1;
        }

        for (idx = 0; idx < x2.nc(); ++idx)
        {
            x2(0, idx) = 100.0;
            v2(0, idx) = 1.0;
        }

        for (idx = 0; idx < x3.nc(); ++idx)
        {
            x3(0, idx) = 100.0;
            v3(0, idx) = 1.0;
        }

        for (idx = 0; idx < x4.nc(); ++idx)
        {
            x4(0, idx) = 100.0;
            v4(0, idx) = 1.0;
        }

        std::pair<particle, particle> x_lim(particle(-x1,-x2,-x3,-x4), particle(x1,x2,x3,x4));
        std::pair<particle, particle> v_lim(particle(-v1,-v2,-v3,-v4), particle(v1,v2,v3,v4));

        p.init(x_lim, v_lim);
        
        start_time = chrono::system_clock::now();

        //p.run(schwefel);
        p.run(eval_net);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "PSO (" << elapsed_time.count() << ")" << std::endl;


        std::string filename = "gbest.dat";
        dlib::serialize(filename) << p.G;



        //std::cin.ignore();
        
        bp = 3;
        

        //resnet_type net_101;

        //std::cout << "net_101" << std::endl;
        //std::cout << net_101 << std::endl;
/*
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

        // This does the setting
        // dlib::set_learning_rate<start_layer, end_layer>(net_name, r1, r2);
        dlib::set_learning_rate<6, 145>(net_34_2, r1, r2);

        // print out the net just to show that the multipliers have changed
        std::cout << "net 34 v2" << std::endl;
        std::cout << net_34_2 << std::endl;
*/
        //while (1);
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

