#define _CRT_SECURE_NO_WARNINGS
//#define _USE_MATH_DEFINES

//#include <yaml-cpp/yaml.h>
//#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>

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


// custom includes
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "file_ops.h"
#include "modulo.h"
//#include "console_input.h"


#define M_PI 3.14159265358979323846
#define M_2PI 6.283185307179586476925286766559

// -------------------------------GLOBALS--------------------------------------
std::string platform;


volatile bool entry = false;
volatile bool run = true;
std::string console_input1;

template<typename T>
inline void vector_to_pair(std::vector<T> &v1, std::vector<T> &v2, std::vector<std::pair<T, T>> &p1)
{
    assert(v1.size() == v2.size());

    uint64_t idx;

    p1.clear();

    for (idx = 0; idx < v1.size(); ++idx)
    {
        p1.push_back(std::make_pair(v1[idx], v2[idx]));
    }

}   // end of vector_to_pair

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx=0, jdx=0;
    uint64_t r, c;
    char key = 0;

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);


    std::string data_directory;
    std::string train_inputfile, test_inputfile;

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
        std::string scenario;
        std::string train_file, test_file;
        std::vector<uint32_t> f;
        double duration;
        uint32_t steps;
        uint32_t h, w;

        int bp = 0;
        std::string input_file = "../../dfd_input_laptop.yml";

        std::ifstream t(input_file);
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string contents = buffer.str();

        std::cout << contents << std::endl;

        ryml::Tree config = ryml::parse_in_arena(ryml::to_csubstr(contents));

        config["scenario"] >> scenario;

        config["train_file"] >> train_file;
        config["test_file"] >> test_file;

        ryml::NodeRef stop_criteria = config["stop_criteria"];
        stop_criteria["hours"] >> duration;
        stop_criteria["steps"] >> steps;

        ryml::NodeRef crop_sizes = config["crop_size"];
        crop_sizes["height"] >> h;
        crop_sizes["width"] >> w;

        auto temp = config["filter_num"].is_seq();
        config["filter_num"] >> f;

        uint8_t bg_dm_value;
        double bg_prob;
        std::vector<std::pair<uint8_t, uint8_t>> bg_br_table;
        std::vector<uint8_t> bg_table_br1, bg_table_br2;

        ryml::NodeRef background = config["background"];
        background["value"] >> bg_dm_value;
        background["probability"] >> bg_prob;
        background["blur_radius1"] >> bg_table_br1;
        background["blur_radius2"] >> bg_table_br2;

        vector_to_pair(bg_table_br1, bg_table_br2, bg_br_table);

        ryml::NodeRef turbulence_parameters = config["turbulence_parameters"];
        auto tp1 = turbulence_parameters["Cn2"];

        double cn2_min, cn2_max;
        tp1["min"] >> cn2_min;
        tp1["max"] >> cn2_max;

        bp = 1;

        //std::ifstream fin(input_file);
        //if (!fin)
        //    std::cout << "bad file" << std::endl;

        //YAML::Node config = YAML::Load(fin);
        //YAML::Node config2 = YAML::LoadFile(input_file);

        //auto scenario = config["scenario"].as<std::string>();
        //auto test2 = config["stop_criteria"][0];

        //double duration = config["stop_criteria"][0]["hours"].as<double>();// = test2[0].as<double>(); //test2["hours"].as<double>();
        //uint64_t steps = config["stop_criteria"][0]["steps"].as<uint64_t>();// = test2[1].as<uint64_t>();  //test2["steps"].as<uint64_t>();

        //std::string train_file2 = config["train_file"].as<std::string>();
        //std::string test_file2 = config["test_file"].as<std::string>();

        ////auto test3 = config["crop_size"];

        //uint32_t h = config["crop_size"][0]["height"].as<uint32_t>();
        //uint32_t w = config["crop_size"][0]["width"].as<uint32_t>();


        //
        //auto test4 = config["filter_num"];

        //auto sz = test4.size();


        //for (uint32_t i = 0; i < sz; ++i)
        //{
        //    f.push_back(test4[i].as<uint32_t>());
        //}
        //bp = 0;



        // ----------------------------------------------------------------------------------------


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

