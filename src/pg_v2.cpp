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


using mnist_net_type = dlib::loss_multiclass_log<
    dlib::fc<10,
    dlib::prelu<dlib::fc<84,
    dlib::prelu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::prelu<dlib::con<6, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

template <typename img_type1>
void check_matrix(img_type1 img)
{
    if (dlib::is_matrix<img_type1>::value == true)
        std::cout << "matrix" << std::endl;
}

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
    std::vector<dlib::matrix<unsigned char>> training_images;
    std::vector<dlib::matrix<unsigned char>> testing_images;
    std::vector<unsigned long> training_labels;
    std::vector<unsigned long> testing_labels;


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

        data_directory = "D:/Projects/MNIST/data";

        // load the data in using the dlib built in function
        dlib::load_mnist_dataset(data_directory, training_images, training_labels, testing_images, testing_labels);


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

        // setup the input image info
        data_directory = "D:/IUPUI/Test_Data/Middlebury_Images_Third/Art/";
        std::string f_img = "Illum2/Exp1/view1.png";
        std::string d_img = "Illum2/Exp1/view1_lin_0.32_2.88.png";
        std::string dm_img = "disp1.png";

        // load the images
        std::array<dlib::matrix<uint16_t>, img_depth> t, tm;
        dlib::matrix<dlib::rgb_pixel> f, f_tmp, d, d_tmp;
        dlib::matrix<uint16_t> dm_tmp, dm, mask;

        dlib::load_image(f_tmp, (data_directory+f_img));
        dlib::load_image(d_tmp, (data_directory+d_img));
        dlib::load_image(dm_tmp, (data_directory + dm_img));

        //split_channels(f_tmp, t);

        // crop the images to the right network size
        // get image size
        uint64_t rows = 368;// crop_sizes[1].first;
        uint64_t cols = 400;// crop_sizes[1].second;

        f.set_size(rows, cols);
        d.set_size(rows, cols);
        dm.set_size(rows, cols);

        // crop the image to fit into the net
        dlib::set_subm(f, 0, 0, rows, cols) = dlib::subm(f_tmp, 0, 0, rows, cols);
        dlib::set_subm(d, 0, 0, rows, cols) = dlib::subm(d_tmp, 0, 0, rows, cols);
        dlib::set_subm(dm, 0, 0, rows, cols) = dlib::subm(dm_tmp, 0, 0, rows, cols);

        // split the channels and combine
        split_channels(f, 0, t);
        split_channels(d, 3, t);

        // test the mask creation
        create_mask(dm, mask, 140, 150);

        // test the mask overlay feature
        apply_mask(t, tm, mask, 0);




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
        save_location = "D:/IUPUI/PhD/Results/dfd_dnn/dnn_reduction/v14v/";
        save_name = "net_v14v_";

        dlib::matrix<uint16_t> map = net(t);

        //gorgon_capture<50> gc_01(net);
        //gc_01.init((save_location + save_name + "art_L50"));
        //gc_01.save_net_output(net);
        //gc_01.close_stream();

        //gorgon_capture<46> gc_02(net);
        //gc_02.init((save_location + save_name + "art_L46"));
        //gc_02.save_net_output(net);
        //gc_02.close_stream();

        //gorgon_capture<45> gc_03(net);
        //gc_03.init((save_location + save_name + "art_L45"));
        //gc_03.save_net_output(net);
        //gc_03.close_stream();

        gorgon_capture<42> gc_04(net);
        gc_04.init((save_location + save_name + "art_L42"));
        gc_04.save_net_output(net);
        gc_04.close_stream();

        gorgon_capture<38> gc_05(net);
        gc_05.init((save_location + save_name + "art_L38"));
        gc_05.save_net_output(net);
        gc_05.close_stream();

        gorgon_capture<37> gc_06(net);
        gc_06.init((save_location + save_name + "art_L37"));
        gc_06.save_net_output(net);
        gc_06.close_stream();

        gorgon_capture<34> gc_07(net);
        gc_07.init((save_location + save_name + "art_L34"));
        gc_07.save_net_output(net);
        gc_07.close_stream();

        gorgon_capture<30> gc_08(net);
        gc_08.init((save_location + save_name + "art_L30"));
        gc_08.save_net_output(net);
        gc_08.close_stream();

        gorgon_capture<29> gc_09(net);
        gc_09.init((save_location + save_name + "art_L29"));
        gc_09.save_net_output(net);
        gc_09.close_stream();

        gorgon_capture<27> gc_10(net);
        gc_10.init((save_location + save_name + "art_L27"));
        gc_10.save_net_output(net);
        gc_10.close_stream();

        gorgon_capture<22> gc_11(net);
        gc_11.init((save_location + save_name + "art_L22"));
        gc_11.save_net_output(net);
        gc_11.close_stream();

        gorgon_capture<18> gc_12(net);
        gc_12.init((save_location + save_name + "art_L18"));
        gc_12.save_net_output(net);
        gc_12.close_stream();

        gorgon_capture<17> gc_13(net);
        gc_13.init((save_location + save_name + "art_L17"));
        gc_13.save_net_output(net);
        gc_13.close_stream();

        gorgon_capture<15> gc_14(net);
        gc_14.init((save_location + save_name + "art_L15"));
        gc_14.save_net_output(net);
        gc_14.close_stream();

        gorgon_capture<13> gc_14a(net);
        gc_14a.init((save_location + save_name + "art_L13"));
        gc_14a.save_net_output(net);
        gc_14a.close_stream();

        gorgon_capture<10> gc_15(net);
        gc_15.init((save_location + save_name + "art_L10"));
        gc_15.save_net_output(net);
        gc_15.close_stream();

        gorgon_capture<6> gc_16(net);
        gc_16.init((save_location + save_name + "art_L06"));
        gc_16.save_net_output(net);
        gc_16.close_stream();

        //gorgon_capture<5> gc_17(net);
        //gc_17.init((save_location + save_name + "art_L05"));
        //gc_17.save_net_output(net);
        //gc_17.close_stream();

        //gorgon_capture<4> gc_17a(net);
        //gc_17a.init((save_location + save_name + "art_L04"));
        //gc_17a.save_net_output(net);
        //gc_17a.close_stream();

        gorgon_capture<2> gc_18(net);
        gc_18.init((save_location + save_name + "art_L02"));
        gc_18.save_net_output(net);
        gc_18.close_stream();

        //gorgon_capture<1> gc_19(net);
        //gc_19.init((save_location + save_name + "art_L01"));
        //gc_19.save_net_output(net);
        //gc_19.close_stream();
 */       
//-----------------------------------------------------------------
// MNIST Net

        //net_name = "D:/Projects/MNIST/nets/mnist_net_05_16_120_84.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_05_15_120_84.dat";
        //net_name = "D:/Projects/MNIST/nets/mnist_net_04_15_120_84.dat";
        net_name = "D:/Projects/MNIST/nets/mnist_net_04_15_75_84_hpc.dat";
        mnist_net_type net;

        // deserialize the network
        dlib::deserialize(net_name) >> net;

        std::cout << net << std::endl;

        save_location = "D:/Projects/MNIST/results/net_04_15_75_84_h/";
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

            gorgon_capture<9> gc_1a(net);
            gc_1a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L09"));
            gc_1a.save_net_output(net);
            gc_1a.close_stream();

            gorgon_capture<8> gc_2(net);
            gc_2.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L08"));
            gc_2.save_net_output(net);
            gc_2.close_stream();

            gorgon_capture<6> gc_2a(net);
            gc_2a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L06"));
            gc_2a.save_net_output(net);
            gc_2a.close_stream();

            gorgon_capture<5> gc_3(net);
            gc_3.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L05"));
            gc_3.save_net_output(net);
            gc_3.close_stream();

            gorgon_capture<4> gc_3a(net);
            gc_3a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L04"));
            gc_3a.save_net_output(net);
            gc_3a.close_stream();

            gorgon_capture<3> gc_4(net);
            gc_4.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L03"));
            gc_4.save_net_output(net);
            gc_4.close_stream();

            gorgon_capture<2> gc_4a(net);
            gc_4a.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L02"));
            gc_4a.save_net_output(net);
            gc_4a.close_stream();

            //gorgon_capture<1> gc_5(net);
            //gc_5.init((save_location + number + save_name + num2str(testing_labels[ti[idx]], "%02u_") + "L01"));
            //gc_5.save_net_output(net);
            //gc_5.close_stream();
        }

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

