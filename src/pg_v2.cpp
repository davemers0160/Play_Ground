#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
//#include <windows.h>
//#include "win_network_fcns.h"
#endif

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
#include <type_traits>

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

// Network includes
#include "dfd_net_v14.h"
//#include "dfd_net_v14_pso_01.h"
//#include "dfd_net_rw_v19.h"
//#include "load_dfd_rw_data.h"
//#include "load_dfd_data.h"

//#include "cyclic_analysis.h"

using namespace std;

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t img_depth;
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

using mnist_net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::prelu<dlib::fc<84,
    dlib::prelu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<6, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>>>;

//using mnist_net_type = dlib::loss_multiclass_log<
//    dlib::fc<10,
//    dlib::htan<dlib::fc<84,
//    dlib::sig<dlib::con<120, 5, 5, 1, 1,
//    dlib::sig<dlib::max_pool<2, 2, 2, 2, dlib::con<16, 5, 5, 1, 1,
//    dlib::sig<dlib::max_pool<2, 2, 2, 2, dlib::con<6, 5, 5, 1, 1,
//    dlib::input<dlib::matrix<unsigned char>>
//    >>>>>>>>>>>>;


using test_net_type = dlib::loss_multiclass_log_per_pixel <
    cbp3_blk<256, 
    dlib::affine<
    dlib::input<std::array<dlib::matrix<uint16_t>, img_depth>>
    >>>;

//----------------------------------------------------------------------------------

template <typename net_type>
dlib::matrix<double, 1, 3> eval_mnist_performance(net_type &net, std::vector<dlib::matrix<unsigned char>> input_images, std::vector<unsigned long> input_labels)
{
    std::vector<unsigned long> predicted_labels = net(input_images);
    int num_right = 0;
    int num_wrong = 0;
    // And then let's see if it classified them correctly.
    for (size_t i = 0; i < input_images.size(); ++i)
    {
        if (predicted_labels[i] == input_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    // std::cout << "training num_right: " << num_right << std::endl;
    // std::cout << "training num_wrong: " << num_wrong << std::endl;
    // std::cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << std::endl;

    dlib::matrix<double, 1, 3> results;
    results = (double)num_right, (double)num_wrong, (double)num_right / (double)(num_right + num_wrong);

    return results;

}   // end of eval_net_performance



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

//template<int from, int to, typename net_type1, typename net_type2>
//typename std::enable_if<from == to>::type
//copy_net(net_type1 const&, net_type2&)
//{
//}

template<int from, int to, typename net_type1, typename net_type2>
//typename std::enable_if<from >= to>::type
void copy_net(net_type1 &in, net_type2 &out)
{
    dlib::layer<to>(out).layer_details() = dlib::layer<from>(in).layer_details();
    //copy_net<from + 1, to>(in, out);
}

template<int from, int to, typename net_type1, typename net_type2>
void copy_layer(net_type1 &in, net_type2 &out, uint64_t size)
{
    auto &layer_params_from = dlib::layer<from>(in).layer_details().get_layer_params();
    float *lp_data_from = layer_params_from.host();


    auto &layer_params_to = dlib::layer<to>(out).layer_details().get_layer_params();
    float *lp_data_to = layer_params_to.host();

    lp_data_to = lp_data_from;

    //for (uint64_t idx = 0; idx < size; ++idx)
    //{
    //    tmp2[idx] = tmp1[idx];
    //}
}

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

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using level1 = ares<512, ares<512, ares_down<512, SUBNET>>>;
template <typename SUBNET> using level2 = ares<256, ares<256, ares<256, ares<256, ares<256, ares_down<256, SUBNET>>>>>>;
template <typename SUBNET> using level3 = ares<128, ares<128, ares<128, ares_down<128, SUBNET>>>>;
template <typename SUBNET> using level4 = ares<64, ares<64, ares<64, SUBNET>>>;

using anet_type = dlib::loss_multiclass_log< dlib::fc<1000, dlib::avg_pool_everything<
    level1<
    level2<
    level3<
    level4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    dlib::input_rgb_image_sized<227>
    >>>>>>>>>>>;

using anet_type2 = dlib::loss_mmod<dlib::con<1,9,9,1,1,
    level1<
    level2<
    level3<
    level4<
    dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<64, 7, 7, 2, 2,
    dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>
    >>>>>>>>>>;



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
    getPlatform(platform);
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


        const char open = '{';
        const char close = '}';
        std::vector<std::string> p1,p2,p3, p4;
        
        std::string line1 = "abc, def, {1,2,3,4,5,6,dog}, {7,8,9,10,11,12,cat}, other";
        std::string line2 = "{1,2,3,4,5,6,dog}, {7,8,9,10,11,12,cat}, other";
        std::string line3 = "abc, def, {1,2,3,4,5,6,dog}, {7,8,9,10,11,12,cat}";
        std::string line4 = "abc, def, other";

        std::string file_name = "D:/Projects/Play_Ground/poly_example.txt";

        //parseCSVFile(file_name, training_file);

        parse_group_line(line1, open, close, p1);
        parse_group_line(line2, open, close, p2);
        parse_group_line(line3, open, close, p3);
        parse_group_line(line4, open, close, p4);


        std::vector<std::vector<std::string>> params;
        parse_group_csv_file(file_name, open, close, params);


        // parse the second param because it has a polygon
        idx = 4;

        // this gets the file names
        std::string f1 = params[idx][0];
        std::string f2 = params[idx][1];

        uint64_t left, right, top, bottom;

        // this now reads the label info
        for (jdx = 2; jdx < params[idx].size(); ++jdx)
        {
            std::vector<std::string> label_info;

            parseCSVLine(params[idx][jdx], label_info);

            // get the label since it is the last element and the remove
            std::string label_name = label_info.back();
            label_info.pop_back();

            // convert the strings to uints
            std::vector<uint32_t> points(label_info.size());
            for (uint32_t kdx = 0; kdx < label_info.size(); ++kdx)
            {
                points[kdx] = (uint32_t)std::stoi(label_info[kdx]);
            }

            // check the size of points.  If there are more than 4 points then the input is
            // a polygon otherwise it is a rectangle
            if (points.size() > 4)
            {
                // now assume that there are and equal number of x,y points
                uint32_t div = points.size() >> 1;

                const auto x = std::minmax_element(begin(points), begin(points) + div);
                const auto y = std::minmax_element(begin(points) + div, end(points));

                left = *x.first;
                right = *x.second;
                top = *y.first;
                bottom = *y.second;
            }
            else
            {
                // create the rect from the x,y, w,h points
                left = points[0];
                top = points[1];
                right = left + points[2];
                bottom = top + points[3];
                
            }

            dlib::rectangle r(left, top, right, bottom);
            dlib::mmod_rect m_r(r, 0.0, label_name);

        }



        bp = 2;

        //uint32_t l1 = 0; //(uint32_t)line.find(open);
        //uint32_t l2 = 0;  //(uint32_t)line.find(close);

        // parse the lines - find the first instance of a group and then the last one
        // then separate the two sections
        //uint32_t g_start = (uint32_t)line.find(open);
        //uint32_t g_stop = (uint32_t)line.rfind(close);

        //// get the group substring
        //std::string sec_start = line.substr(0, g_start);
        //std::string group = line.substr(g_start, g_stop - g_start + 1);
        //std::string sec_end = line.substr(g_stop+1, line.length()-1);

        //parse_line(sec_start, ',', params);
        //parse_line(sec_end, ',', params);

        //while (l2 < group.size())
        //{
        //    l1 = (uint32_t)group.find(open);
        //    l2 = (uint32_t)group.find(close);
        //    std::string g = group.substr(l1 + 1, l2 - l1 - 1);
        //    trim(g);
        //    if (g.size() > 0)
        //    {
        //        group_params.push_back(g);
        //    }
        //}

        //stringstream gs(group);
        //while (gs.good())
        //{
        //    std::string s1;
        //    std::string s2;
        //    std::getline(gs, s1, open);
        //    std::getline(gs, s2, close);

        //    trim(s2);
        //    if (s2.size() > 0)
        //    {
        //        group_params.push_back(s2);
        //    }
        //}




/*
        std::vector<std::string> labels;
        anet_type net_34;
        dlib::deserialize("../nets/resnet34_1000_imagenet_classifier.dnn") >> net_34 >> labels;

        std::cout << net_34 << std::endl;

        anet_type2 net_34_2;

        bp = 2;
       
        std::cout << net_34_2 << std::endl;

        //float* tmp1 = dlib::layer<6>(net_34).layer_details().get_layer_params().host();
        //float* tmp2 = dlib::layer<6>(net_34_2).layer_details().get_layer_params().host();

        //std::vector<float> pd(tmp1, tmp1 + 512);
        //tmp2 = pd.data();

        auto &ld11 = dlib::layer<138>(net_34).layer_details();
        auto &ld21 = dlib::layer<137>(net_34_2).layer_details();

        copy_net<138, 137>(net_34, net_34_2);

        auto &ld12 = dlib::layer<138>(net_34).layer_details();
        auto &ld22 = dlib::layer<137>(net_34_2).layer_details();

        auto& tmp1 = dlib::layer<138>(net_34).layer_details().get_layer_params();
        auto& tmp2 = dlib::layer<137>(net_34_2).layer_details().get_layer_params();

        dlib::dnn_trainer<anet_type2, dlib::sgd> trainer(net_34_2, dlib::sgd());
        //trainer.train_one_step(NULL, NULL);

        //std::cout << net_34_2.subnet();

        //net_34_2.subnet().subnet() = net_34.subnet().subnet().subnet();



        //copy_layer<137, 138>(net_34, net_34_2, 512);

        //tmp2 = dlib::layer<6>(net_34_2).layer_details().get_layer_params().host();

        //dlib::layer<2>(net_34_2) = dlib::layer<3>(net_34);
        ////dlib::layer
        //
        //std::cout << net_34_2 << std::endl;

        //auto &layer_params = dlib::layer<5>(net_34_2).layer_details().get_layer_params();
        //const float* params_data = layer_params.host();

        //auto &layer_params2 = dlib::layer<6>(net_34).layer_details().get_layer_params();
        //const float* params_data2 = layer_params2.host();


        //auto &t2 = dlib::layer<143>(net_34_2);   // = dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>;
        ////dlib::layer<143>(net_34_2) = dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>;


        //----------------------------------------------------------------
        //test_net_type tnet;

        //std::cout << tnet << std::endl;

        //auto& af_details = dlib::layer<4>(tnet).layer_details();

*/

        //----------------------------------------------------------------
        data_directory = "D:/Projects/MNIST/data";

        // load the data in using the dlib built in function
        //dlib::load_mnist_dataset(data_directory, training_images, training_labels, testing_images, testing_labels);


        // get the location of the network
        std::string net_name;
        //net_name = "D:/IUPUI/PhD/Results/dfd_dnn/dnn_reduction/2704823.pbs01_0_nets/nets/dfd_net_v14v_61_U_32_HPC.dat";
        //net_name = "D:/IUPUI/PhD/Results/dfd_dnn/2649400.pbs01_0_nets/nets/dfd_net_v14f_61_U_32_HPC.dat";
        //net_name = "D:/IUPUI/PhD/Results/dfd_dnn/2651620.pbs01_0_nets/nets/dfd_net_v14i_61_U_32_HPC.dat";
        //net_name = "D:/IUPUI/PhD/Results/dfd_dnn/dnn_reduction/2646754.pbs01_0_nets/nets/dfd_net_v14e_61_U_32_HPC.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_L05_100.dat";

        //declare the network
        //dfd_net_type net;
        //mnist_net_type net;

        //// deserialize the network
        //dlib::deserialize(net_name) >> net;

        //std::cout << net << std::endl;

        bp = 1;

        //----------------------------------------------------------------
/*
        DataLogStream.open("random_cropper_selection.txt", ios::out | ios::app);

        train_inputfile = "D:/IUPUI/DfD/DfD_DNN/dfd_train_data_sm2.txt";

        // parse through the supplied training csv file
        parseCSVFile(train_inputfile, training_file);

        // the first line in this file is now the data directory
        data_directory = training_file[0][0];
        training_file.erase(training_file.begin());

        std::cout << "Loading training images..." << std::endl;

        start_time = chrono::system_clock::now();
        loadData(training_file, data_directory, tr, gt_train, tr_image_files);
        stop_time = chrono::system_clock::now();

        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        std::cout << "Loaded " << tr.size() << " training image sets in " << elapsed_time.count() / 60 << " minutes." << std::endl << std::endl;

        uint32_t crop_num = 32;
        uint64_t img_index = 0;
        
        for (uint32_t kdx = 0; kdx < 63500; ++kdx)
        {
            for (uint32_t mdx = 0; mdx < crop_num; ++mdx)
            {
                img_index = rnd.get_integer(tr.size());
                make_random_cropping_rect(tr[img_index][0], rect_im);
                DataLogStream << img_index << "," << tr[img_index][0].nr() << "," << tr[img_index][0].nc() << "," << rect_im.left() << "," << rect_im.top() << std::endl;
            }
        }

        DataLogStream.close();

        bp = 3;
        return 0;
        */
        //-------------------------------------------------------------------

        // setup the input image info
        //data_directory = "D:/IUPUI/Test_Data/Middlebury_Images_Third/Art/";
        //std::string f_img = "Illum2/Exp1/view1.png";
        //std::string d_img = "Illum2/Exp1/view1_lin_0.32_2.88.png";
        //std::string dm_img = "disp1.png";

        //// load the images
        //std::array<dlib::matrix<uint16_t>, img_depth> t, tm;
        //dlib::matrix<dlib::rgb_pixel> f, f_tmp, d, d_tmp;
        //dlib::matrix<uint16_t> dm_tmp, dm, mask;

        //dlib::load_image(f_tmp, (data_directory+f_img));
        //dlib::load_image(d_tmp, (data_directory+d_img));
        //dlib::load_image(dm_tmp, (data_directory + dm_img));

        ////split_channels(f_tmp, t);

        //// crop the images to the right network size
        //// get image size
        //uint64_t rows = 368;// crop_sizes[1].first;
        //uint64_t cols = 400;// crop_sizes[1].second;

        //f.set_size(rows, cols);
        //d.set_size(rows, cols);
        //dm.set_size(rows, cols);

        //// crop the image to fit into the net
        //dlib::set_subm(f, 0, 0, rows, cols) = dlib::subm(f_tmp, 0, 0, rows, cols);
        //dlib::set_subm(d, 0, 0, rows, cols) = dlib::subm(d_tmp, 0, 0, rows, cols);
        //dlib::set_subm(dm, 0, 0, rows, cols) = dlib::subm(dm_tmp, 0, 0, rows, cols);

        //// split the channels and combine
        //split_channels(f, 0, t);
        //split_channels(d, 3, t);

        //// test the mask creation
        //create_mask(dm, mask, 140, 150);

        //// test the mask overlay feature
        //apply_mask(t, tm, mask, 0);




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

        std::string save_location;
        std::string save_name;

//-----------------------------------------------------------------
// DFD Net
/*
        
        std::cout << "Loading test images..." << std::endl;

        test_inputfile = "D:/IUPUI/DfD/DfD_DNN/dfd_test_data_sm2.txt";
        parseCSVFile(test_inputfile, test_file);
        data_directory = test_file[0][0];
        test_file.erase(test_file.begin());

        start_time = chrono::system_clock::now();
        loadData(test_file, data_directory, te, gt_test, te_image_files);
        stop_time = chrono::system_clock::now();

        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
        std::cout << "Loaded " << te.size() << " test image sets in " << elapsed_time.count() / 60.0 << " minutes." << std::endl << std::endl;

        std::string net_version = "v14a";

        save_location = "D:/IUPUI/PhD/Results/dfd_dnn/dnn_reduction/" + net_version + "/";
        save_name = "net_" + net_version + "_";
        net_name = "D:/IUPUI/PhD/Results/dfd_dnn/dnn_reduction/" + net_version + "/nets/dfd_net_v14a_61_U_32_HPC.dat";

        dfd_net_type net;

        // deserialize the network
        dlib::deserialize(net_name) >> net;

        std::cout << net << std::endl;

        std::vector<std::string> data_name = { "art","books","reindeer" };
        std::vector<uint32_t> ti = { 4, 22, 40 };

        for (idx = 0; idx < ti.size(); ++idx)
        {
            std::cout << "Running: " << data_name[idx] << std::endl;
            make_dir(save_location, data_name[idx]);

            dlib::matrix<uint16_t> map = net(te[ti[idx]]);
/*
            gorgon_capture<50> gc_01(net);
            gc_01.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L50"));
            gc_01.save_net_output(net);
            gc_01.close_stream();

            gorgon_capture<46> gc_02(net);
            gc_02.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L46"));
            gc_02.save_net_output(net);
            gc_02.close_stream();

            gorgon_capture<44> gc_03(net);
            gc_03.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L44"));
            gc_03.save_net_output(net);
            gc_03.close_stream();

            gorgon_capture<42> gc_04(net);
            gc_04.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L42"));
            gc_04.save_net_output(net);
            gc_04.close_stream();

            gorgon_capture<38> gc_05(net);
            gc_05.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L38"));
            gc_05.save_net_output(net);
            gc_05.close_stream();

            gorgon_capture<36> gc_06(net);
            gc_06.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L36"));
            gc_06.save_net_output(net);
            gc_06.close_stream();

            gorgon_capture<34> gc_07(net);
            gc_07.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L34"));
            gc_07.save_net_output(net);
            gc_07.close_stream();

            gorgon_capture<30> gc_08(net);
            gc_08.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L30"));
            gc_08.save_net_output(net);
            gc_08.close_stream();

            gorgon_capture<28> gc_09(net);
            gc_09.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L28"));
            gc_09.save_net_output(net);
            gc_09.close_stream();

            gorgon_capture<27> gc_10(net);
            gc_10.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L27"));
            gc_10.save_net_output(net);
            gc_10.close_stream();

            //gorgon_capture<25> gc_10a(net);
            //gc_10a.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L25"));
            //gc_10a.save_net_output(net);
            //gc_10a.close_stream();

            gorgon_capture<22> gc_11(net);
            gc_11.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L22"));
            gc_11.save_net_output(net);
            gc_11.close_stream();

            gorgon_capture<18> gc_12(net);
            gc_12.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L18"));
            gc_12.save_net_output(net);
            gc_12.close_stream();

            gorgon_capture<16> gc_13(net);
            gc_13.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L16"));
            gc_13.save_net_output(net);
            gc_13.close_stream();

            gorgon_capture<15> gc_14(net);
            gc_14.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L15"));
            gc_14.save_net_output(net);
            gc_14.close_stream();

            //gorgon_capture<13> gc_14a(net);
            //gc_14a.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L13"));
            //gc_14a.save_net_output(net);
            //gc_14a.close_stream();

            gorgon_capture<10> gc_15(net);
            gc_15.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L10"));
            gc_15.save_net_output(net);
            gc_15.close_stream();

            gorgon_capture<6> gc_16(net);
            gc_16.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L06"));
            gc_16.save_net_output(net);
            gc_16.close_stream();

            gorgon_capture<4> gc_17(net);
            gc_17.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L04"));
            gc_17.save_net_output(net);
            gc_17.close_stream();

            gorgon_capture<2> gc_18(net);
            gc_18.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L02"));
            gc_18.save_net_output(net);
            gc_18.close_stream();
*/
/*
            gorgon_capture<1> gc_19(net);
            gc_19.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L01"));
            gc_19.save_net_output(net);
            gc_19.close_stream();

            //gorgon_capture<0> gc_20(net);
            //gc_20.init((save_location + data_name[idx] + "/" + save_name + data_name[idx] + "_L00"));
            //gc_20.save_net_output(net);
            //gc_20.close_stream();

        }

*/
//-----------------------------------------------------------------
// MNIST Net
/*
        //net_name = "D:/Projects/MNIST/nets/mnist_net_05_16_120_84.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_05_15_120_84.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_04_15_120_84.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_06_16_51_55.dat";
        net_name = "D:/Projects/MNIST/nets/mnist_net_04_13_68_55.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_v2_06_16_120_61.dat";
        mnist_net_type net;

        // deserialize the network
        dlib::deserialize(net_name) >> net;

        std::cout << net << std::endl;
        dlib::matrix<double, 1, 3> training_results = eval_mnist_performance(net, training_images, training_labels);
        //auto results = net(training_images);

        double avg_train_time = 0.0;
        for (uint32_t idx = 0; idx < 30; ++idx)
        {
            start_time = chrono::system_clock::now();
            //training_results = eval_net_performance(test_net, training_images, training_labels);
            net(training_images);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);
            avg_train_time += elapsed_time.count();
            std::cout << ".";
        }
        avg_train_time = avg_train_time / 30.0;
        std::cout << endl;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Average run time:   " << avg_train_time << std::endl;
        std::cout << "Training num_right: " << training_results(0, 0) << std::endl;
        std::cout << "Training num_wrong: " << training_results(0, 1) << std::endl;
        std::cout << "Training accuracy:  " << training_results(0, 2) << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        save_location = "D:/Projects/MNIST/results/net_04_15_072_84/";
        save_name = "net_out_";
        std::vector<uint32_t> ti = { 0,1,2,3,4,7,8,11,18,61 };//   7, 2, 1, 0, 4, 9, 5, 6, 3, 8

        for (idx = 0; idx < ti.size(); ++idx)
        {
            std::cout << "Running: " << testing_labels[ti[idx]] << std::endl;

            // run the image through the network
            unsigned long predicted_labels = net(testing_images[ti[idx]]);
            std::string number = num2str(testing_labels[ti[idx]], "%02u/");
            make_dir(save_location, number);

            gorgon_capture<11> gc_1(net);
            gc_1.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L11"));
            gc_1.save_net_output(net);
            gc_1.close_stream();

            //v2
            //gorgon_capture<10> gc_1a(net);
            //gc_1a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L10"));
            //gc_1a.save_net_output(net);
            //gc_1a.close_stream();

            //v1
            gorgon_capture<9> gc_1a(net);
            gc_1a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L09"));
            gc_1a.save_net_output(net);
            gc_1a.close_stream();

            gorgon_capture<8> gc_2(net);
            gc_2.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L08"));
            gc_2.save_net_output(net);
            gc_2.close_stream();

            //v2
            //gorgon_capture<7> gc_2a(net);
            //gc_2a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L07"));
            //gc_2a.save_net_output(net);
            //gc_2a.close_stream();

            gorgon_capture<6> gc_3(net);
            gc_3.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L06"));
            gc_3.save_net_output(net);
            gc_3.close_stream();

            //v1
            gorgon_capture<5> gc_3b(net);
            gc_3b.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L05"));
            gc_3b.save_net_output(net);
            gc_3b.close_stream();

            gorgon_capture<4> gc_3a(net);
            gc_3a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L04"));
            gc_3a.save_net_output(net);
            gc_3a.close_stream();

            //v1
            gorgon_capture<3> gc_4(net);
            gc_4.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L03"));
            gc_4.save_net_output(net);
            gc_4.close_stream();

            gorgon_capture<2> gc_4a(net);
            gc_4a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L02"));
            gc_4a.save_net_output(net);
            gc_4a.close_stream();

        }
*/
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

