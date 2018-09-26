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
#include "getCurrentTime.h"
#include "get_platform.h"
//#include "pso.h"
#include "ycrcb_pixel.h"
#include "num2string.h"
#include "file_parser.h"
#include "load_dfd_data.h"
#include "dfd_array_cropper.h"
#include "rot_90.h"
#include "read_binary_image.h" 
#include "make_dir.h"

#include "dlib_elu.h"

//#include "dfd_net_v8d.h"

using namespace std;

// ----------------------------------------------------------------------------------------


void write_xml(std::string filename)
{
    std::ofstream xmlStream;

    uint64_t l = 1, n = 2, nr = 3, nc = 3, k = 2;

    try
    {
        xmlStream.open(filename, ios::out);
        xmlStream << "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
        xmlStream << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n";
        xmlStream << "<gorgon_data>\n";
        xmlStream << "    <version major='3' minor='0' />\n";
        xmlStream << "    <layer number='" << l << "'/>\n"; 
        xmlStream << "    <filter n='" << n << "' rows='" << nr << "' cols='" << nc << "' k='" << k << "'/>\n";
        xmlStream << "</gorgon_data>\n\n";
        xmlStream.close();

    }
    catch (std::exception &e)
    {
        std::cout << "Error saving XML metadata..." << std::endl;
        std::cout << e.what() << std::endl;
    }
}   // end of write_xml

// ----------------------------------------------------------------------------------------

double schwefel(dlib::matrix<double> x)
{
	// f(x_n) = -418.982887272433799807913601398*n = -837.965774544867599615827202796
    // x_n = 420.968746

	double result = 0.0;
	
	for(uint64_t idx=0; idx<x.nr(); ++idx)
	{
		result += -x(idx,0)*std::sin(std::sqrt(std::abs(x(idx,0))));
	}
	
	return result;
	
}	// end of schwefel



// ----------------------------------------------------------------------------------------
//
//void get_mat(uint32_t &height, uint32_t &width, dlib::matrix<float> &dimg)
//{
//
//    float *d2;
//
//    std::string file_name = "D:/Projects/MATAS/data/training/arresting/3ITYIHDWlPM_38.resnet18.bin";
//
//    read_binary_image(file_name, width, height, d2);
//    dimg = dlib::mat(d2, height, width);
//
//}


// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx, jdx;

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    unsigned long training_duration = 1;  // number of minutes to train 
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::vector<std::vector<std::string>> training_file;
    std::string data_directory;

    std::string platform;
    getPlatform(platform);
    std::cout << "Platform: " << platform << std::endl;

    if (platform.compare(0, 6, "Laptop") == 0)
    {
        std::cout << "Match!" << std::endl;
    }

    try
    {
        std::cout << "Input: " << argv[1] << std::endl;

        int bp = 0;

        //std::string image_location = "D:/IUPUI/Test_Data/Middlebury_Images_Third/Aloe/";
        //std::string filename = "disp1.png";

// ----------------------------------------------------------------------------------------
        //std::vector<std::string> data;
        //std::string lpMsgBuf;
        //std::string console_input;
        //std::string address;

        //get_ip_address(data, lpMsgBuf);

        //for (idx = 0; idx < data.size(); ++idx)
        //{
        //    std::cout << "IP Address [" << idx << "]: " << data[idx] << std::endl;
        //}

        //std::cout << "Select IP Address: ";
        //std::getline(std::cin, console_input);
        //address = data[stoi(console_input)];


        //bp = 1;
        //// ----------------------------------------------------------------------------------------


        //std::string rx_message = "{\"prod_line\": \"OS-1-64\", \"prod_pn\": \"840-101396-02\", \"prod_sn\": \"991805000142\", \"base_pn\": \"000-101323-01\", \"base_sn\": \"11E0211\", \"image_rev\": \"ousteros-image-prod-aries-v1.2.0-201804232039\", \"build_rev\": \"v1.2.0\", \"proto_rev\": \"v1.1.0\", \"build_date\": \"2018-05-02T18:37:13Z\", \"status\": \"RUNNING\"}";

        //std::vector<std::string> lidar_info;
        //lidar_info.clear();

        //std::vector<std::string> params, params2;
        //parseCSVLine(rx_message, params);
        //for (uint32_t idx = 0; idx < params.size()-1; ++idx)
        //{
        //    parse_line(params[idx], ':', params2);
        //    std::string info = params2[1];
        //    lidar_info.push_back(info.substr(1, info.length() - 2));
        //}
        //bp = 2;

        // ----------------------------------------------------------------------------------------


        // MNIST is broken into two parts, a training set of 60000 images and a test set of
        // 10000 images.  Each image is labeled so that we know what hand written digit is
        // depicted.  These next statements load the dataset into memory.
        std::vector<dlib::matrix<unsigned char>> training_images;
        std::vector<unsigned long>         training_labels;
        std::vector<dlib::matrix<unsigned char>> testing_images;
        std::vector<unsigned long>         testing_labels;
        dlib::load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);

        using net_type = dlib::loss_multiclass_log<
            dlib::fc<10,
            dlib::elu<dlib::fc<84,
            dlib::relu<dlib::fc<120,
            dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
            dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
            dlib::input<dlib::matrix<unsigned char>>
            >>>>>>>>>>>>;

        net_type net;

        dlib::dnn_trainer<net_type> trainer(net);
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_mini_batch_size(128);
        trainer.be_verbose();

        trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));

        trainer.train(training_images, training_labels);

        net.clean();


        std::vector<unsigned long> predicted_labels = net(training_images);
        int num_right = 0;
        int num_wrong = 0;
        // And then let's see if it classified them correctly.
        for (size_t i = 0; i < training_images.size(); ++i)
        {
            if (predicted_labels[i] == training_labels[i])
                ++num_right;
            else
                ++num_wrong;

        }
        cout << "training num_right: " << num_right << endl;
        cout << "training num_wrong: " << num_wrong << endl;
        cout << "training accuracy:  " << num_right / (double)(num_right + num_wrong) << endl;







// ----------------------------------------------------------------------------------------
		// let's start looking at PSO as an option!!!
		// first let's test the schwefel function
		
		//std::vector<double> x={4,4};
		
		//double f = schwefel(x);
		
		//std::cout << "f = " << f << std::endl;

        //uint64_t N = 100;
        //uint64_t itr_max = 50;
        //double c1 = 2.1;           // PSO constant
        //double c2 = 2.0;           // PSO constant
        //double phi = c1 + c2;
        //double K = 2.0 / (std::abs(2.0 - phi - std::sqrt((phi*phi) - (4.0 * phi))));

        //std::vector<std::pair<double, double>> v_lim = { {-5.0, 5.0}, {-5.0, 5.0} };
        //std::vector<std::pair<double, double>> x_lim = { { -500.0, 500.0 },{ -500.0, 500.0 } };
        //dlib::matrix<double> X;
        //std::vector<double> p;
        //double f = 0;
        //uint64_t itr_count = 0;

        //pso_handler pso(N, 2, itr_max, c1, c2);

        //pso.init(x_lim, X);

        // run through first iteration
        // step 1 evaluate X
        //for (itr_count = 0; itr_count < itr_max; ++itr_count)
        //{
        //    for (idx = 0; idx < N; ++idx)
        //    {
        //        f = schwefel(dlib::colm(X, idx));
        //        pso.set_P(f, itr_count, idx);
        //    }

        //    //uint64_t p_idx = dlib::index_of_min(dlib::rowm(P, itr_count));

        //    pso.update(X, itr_count, v_lim, x_lim);
        //    dlib::matrix<double> G_best = pso.get_G_best();

        //    std::cout << "I: " << itr_count << "\tG_best: " << G_best(0,0) << ", " << G_best(1,0);
        //    std::cout << "\tG: " << pso.get_G() << std::endl;

        //}

        // step 2 find best X that minimizes F put into P_Best and P **


        // step 3 compare the best P/P_best with G/G_best


        // step 4 update V


        // step 5 update X


        // step 6 go to 1

       // dlib::matrix<double> P = pso.get_P();


        bp = 1;

        std::pair<uint64_t, std::vector<double>> P_best;
        std::pair<uint64_t, std::vector<double>> G_best;

        // Evaluate the function to determine which population member performed the best
        

        //std::vector<double>::iterator pb = std::min_element(std::begin(P[itr_count]), std::end(P[itr_count]));
        //P_best = std::make_pair((uint64_t)std::distance(std::begin(P[itr_count]), pb), X[P_best.first]);
        //G_best = std::make_pair((uint64_t)std::distance(std::begin(P[itr_count]), pb), X[P_best.first]);
        //P_best.first = (uint64_t)std::distance(std::begin(P[itr_count]), pb);
        //P_best.second = X[P_best.first];

        bp = 2;

        // uint64_t l = 1, n = 2, nr = 3, nc = 3, k = 2;

        // float a = 20.0;
        // float b = -1e-6;

        // uint32_t int_a = reinterpret_cast<uint32_t>(&a);
        // uint32_t int_b = reinterpret_cast<uint32_t>(&b);


        // write_xml("test.xml");

        // uint64_t step = 100;

        // uint64_t data_size = n*nr*nc*k;

        // std::ofstream test_stream;

        // test_stream.open("test_Stream.dat", std::ios::binary);


        // uint32_t magic_number = 0x00FF;

        // test_stream.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));

        // test_stream.write(reinterpret_cast<const char*>(&step), sizeof(step));

        ////dlib::serialize(step, test_stream);

        // for (uint64_t idx = 0; idx < data_size; ++idx)
        // {
            // a = a + static_cast<float>(idx);
            ////dlib::serialize(a, test_stream);
            // test_stream.write(reinterpret_cast<const char*>(&a), sizeof(a));

        // }

        // step = 200;
        // test_stream.write(reinterpret_cast<const char*>(&step), sizeof(step));

        // for (uint64_t idx = 0; idx < data_size; ++idx)
        // {
            // a = a + static_cast<float>(idx);
            ////dlib::serialize(a, test_stream);
            // test_stream.write(reinterpret_cast<const char*>(&a), sizeof(a));
        // }
        // test_stream.close();

        // std::ifstream test_stream1;

        // test_stream1.open("test_Stream.dat", std::ios::in | std::ios::binary);

        ////read in the file
        // uint32_t num;
        // test_stream1.read(reinterpret_cast<char*>(&num), sizeof(num));

        // uint64_t step2;
        // test_stream1.read(reinterpret_cast<char*>(&step2), sizeof(step2));

        // b = 0;
        // for (uint64_t idx = 0; idx < data_size; ++idx)
        // {
            // test_stream1.read(reinterpret_cast<char*>(&b), sizeof(b));
        // }

        ////read in xml file

        // gorgon_param_struct gorgon_params;
        // gorgon_doc_handler gdh(gorgon_params);
        // std::string xml_file = "D:/IUPUI/DfD/DfD_DNN/nets/gorgon_dfd_v8_32_U2_Laptop_l01.xml";
        // uint64_t ext_loc = xml_file.rfind('.');

        // std:string data_file = xml_file.substr(0, ext_loc) + ".dat";

        // dlib::parse_xml(xml_file, gdh);

        ////uint64_t data_size2 = gorgon_params.get_data_size();
        // std::ifstream g_stream;

        // g_stream.open(data_file, std::ios::in | std::ios::binary);
        // std::vector<gorgon_data_struct> gorgon_data;

        // load_params(g_stream, gorgon_params, gorgon_data);

        // g_stream.close();
/*
        mmap::MMap_File map_file;

        map_file.open(map_file_name);

        uint64_t position = 0;
        uint8_t data;

        map_file.read(position, data);
        map_file.read(position, data);
        map_file.read(position, data);
        map_file.read(position, data);
        map_file.read(position, data);
        map_file.read(position, data);
        map_file.read(position, data);
        map_file.read(position, data);

        uint64_t test = 123456789;
        position = 0;

        map_file.write(position, test);

        getCurrentTime(sdate, stime);
        std::cout << "Date: " << sdate << "    Time: " << stime << std::endl;

        //uint8_t *data_in = (uint8_t *)map_file.data();

        //std::vector<uint8_t> data_out(data_in, data_in + 5);


        //data_out[1] = 10;



        //while (elapsed_time.count() / 60 < training_duration)
        //{
        //    stop_time = chrono::system_clock::now();
        //    elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);




        //}

        //getCurrentTime(sdate, stime);
        //std::cout << "Date: " << sdate << "    Time: " << stime << std::endl;
        map_file.close();
        */



    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
        std::cin.ignore();
    }
	return 0;
	
}	// end of main

