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
#include <mutex>

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

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    int bp = 0;

    get_platform(platform);
    std::cout << "Platform: " << platform << std::endl << std::endl;


    try
    {
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

#else
        std::cout << "argv[0]: " << std::string(argv[0]) << std::endl;
        std::string exe_path = get_ubuntu_path();
        std::cout << "Path: " << exe_path << std::endl;
#endif

        //----------------------------------------------------------------------------------------
        // variables
        uint32_t num_threads; 
        uint32_t num_loops;
        uint32_t num_blocks;
        
        
        num_loops = 100;
        //num_blocks = std::ceil(num_loops / (double)num_threads);

        start_time = chrono::system_clock::now();
        num_threads = std::max(1U, std::thread::hardware_concurrency() - 1);
        //{
            std::vector<std::thread> threads(num_threads);
            std::mutex critical;
            for (int t = 0; t < num_threads; t++)
            {
                threads[t] = std::thread(std::bind([&](const int bi, const int ei, const int t)
                {

                    // loop over all items
                    for (int idx = bi; idx < ei; ++idx)
                    {
                        //for (idx = 0; idx < num_loops; ++idx)
                        {
                            Sleep(500);
                            std::lock_guard<std::mutex> lock(critical);
                            std::cout << "Index: " << idx << std::endl;
                        }
                    }
                }, t* num_loops / num_threads, (t + 1) == num_threads ? num_loops : (t + 1) * num_loops / num_threads, t));

            }
            std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });

        //}
        stop_time = chrono::system_clock::now();


        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "elapsed_time " << elapsed_time.count() << std::endl;

        //----------------------------------------------------------------------------------------

        bp = 4;

    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    std::cout << std::endl << "Press Enter to close" << std::endl;
    std::cin.ignore();

	return 0;

}	// end of main

