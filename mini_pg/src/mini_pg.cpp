#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

//#include <yaml-cpp/yaml.h>
//#define RYML_SINGLE_HDR_DEFINE_NOW
//#include <ryml_all.hpp>

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#include <windows.h>
//#include "win_network_fcns.h"
//#include <winsock2.h>
//#include <iphlpapi.h>
//
//#pragma comment(lib, "IPHLPAPI.lib")    // Link with Iphlpapi.lib
#else
#include <dlfcn.h>
typedef void* HINSTANCE;
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
#include <random>
#include <bitset>


// custom includes
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "file_ops.h"
#include "modulo.h"
//#include "console_input.h"

//#include "ocv_threshold_functions.h"

// OpenCV includes
// #include <opencv2/core.hpp>           
// #include <opencv2/highgui.hpp>     
// #include <opencv2/imgproc.hpp> 
// #include <opencv2/video.hpp>
// #include <opencv2/imgcodecs.hpp>

#include "rds.h"
#include "dsp/dsp_windows.h"

#include <test_gen_lib.h>

#define M_PI 3.14159265358979323846
#define M_2PI 6.283185307179586476925286766559f

// -------------------------------GLOBALS--------------------------------------
std::string platform;


volatile bool entry = false;
volatile bool run = true;
std::string console_input1;

//-----------------------------------------------------------------------------
template <typename T>
std::vector<std::complex<float>> generate_fm(std::vector<T> data, uint64_t sample_rate, uint32_t N, float k)
{
    uint64_t idx;

    // interpolation scale multiplier
    //N = math.floor(rf_fs / audio_fs)
    //sample_rate = N * audio_fs
    uint64_t num_data_samples = data.size();
    uint64_t num_rf_samples = std::floor(num_data_samples * N);

    // scale
    std::complex<float> scale = (M_2PI * k) * std::complex<float>(0, M_2PI * k);

    // create empty data
    std::vector<std::complex<float>> iq(num_rf_samples, std::complex<float>(0.0, 0.0));


    // upsample data
    std::vector<float> y2 = upsample_data(data, N);

    // filter data
    int64_t num_taps = 20*N + 1;

    float fc = 1.0 / (float)N;
    std::vector<float> lpf = DSP::create_fir_filter<float>(num_taps, fc, &DSP::blackman_nuttall_window);

    std::vector<float> y2_f;
    apply_filter(y2, lpf, y2_f);

    // shift data to approximately zero mean
    //y2_mean = np.mean(y2)
    //y2 = y2 - y2_mean

    float accum = 0.0f;

    // apply FM modulation
    for(idx=0; idx< y2_f.size(); ++idx)
    {
        //y_accum = np.cumsum(y2)
        accum += y2_f[idx];
        
        //iq = np.exp(scale * y_accum)
        iq[idx] = std::exp(scale * accum);
    }

    return iq;

}   // end of generate_fm


//-----------------------------------------------------------------------------
template<typename T>
void save_complex_data(std::string filename, std::vector<std::complex<T>> data)
{
    std::ofstream data_file;

    //T r, q;

    data_file.open(filename, ios::out | ios::binary);

    if (!data_file.is_open())
    {
        std::cout << "Could not save data. Closing... " << std::endl;
        //std::cin.ignore();
        return;
    }

    data_file.write(reinterpret_cast<const char*>(data.data()), 2 * data.size() * sizeof(T));

    //for (uint64_t idx = 0; idx < data.size(); ++idx)
    //{
    //    //r = data[idx].real();
    //    //q = data[idx].imag();
    //    data_file.write(reinterpret_cast<const char*>(data[idx].real()), sizeof(T));
    //    data_file.write(reinterpret_cast<const char*>(data[idx].imag()), sizeof(T));
    //}
    data_file.close();
}

// ----------------------------------------------------------------------------
void save_complex_data(std::string filename, int16_t* data, uint64_t data_size)
{
    std::ofstream data_file;

    //T r, q;

    data_file.open(filename, ios::out | ios::binary);

    if (!data_file.is_open())
    {
        std::cout << "Could not save data. Closing... " << std::endl;
        //std::cin.ignore();
        return;
    }

    data_file.write(reinterpret_cast<const char*>(data), data_size * sizeof(*data));

    data_file.close();
}

/*
//----------------------------------------------------------------------------
inline void get_rect(std::vector<cv::Point>& p, cv::Rect& r, int64_t img_w, int64_t img_h, int64_t x_padding = 40, int64_t y_padding = 40)
{
    uint64_t idx;
    int64_t min_x = LLONG_MAX, min_y = LLONG_MAX;
    int64_t max_x = 0, max_y = 0;

    for (idx = 0; idx < p.size(); ++idx)
    {
        min_x = std::min(min_x, (int64_t)p[idx].x);
        min_y = std::min(min_y, (int64_t)p[idx].y);
        max_x = std::max(max_x, (int64_t)p[idx].x);
        max_y = std::max(max_y, (int64_t)p[idx].y);

    }

    min_x = std::max(0LL, min_x - x_padding);
    max_x = std::min(img_w, max_x + x_padding);

    min_y = std::max(0LL, min_y - y_padding);
    max_y = std::min(img_h, max_y + y_padding);

    r = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);

}   // end of get_rect

*/
//----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
inline std::vector<uint32_t> gray_code(uint16_t num_bits)
{
    std::vector<uint32_t> gc;

    uint32_t idx;

    uint32_t num = 1 << (num_bits);

    for(idx=0; idx<num; ++idx)
    {
        gc.push_back(idx ^ (idx >> 1));
    }

    return gc;
}

//-----------------------------------------------------------------------------
inline std::vector<std::complex<float>> generate_qam_constellation(uint16_t num_bits)
{
    uint32_t idx, jdx;

    uint32_t side_length = 1 << (num_bits >> 1);
    double step = 2.0;
    int16_t start = -side_length + 1;
    float scale = 1.0/abs(start);
    uint32_t index = 0;

    std::vector<uint32_t> gc = gray_code(num_bits);
    std::vector<std::complex<float>> bit_mapper(1 << num_bits);

    // create the base locations for the constellation
    std::vector<float> c_p(side_length, 0);

    // create the primary normalized points for the constellation
    for (idx = 0; idx < side_length; ++idx)
    {
        c_p[idx] = (start * scale);
        start += step;
    }

    // cycle through the side_length x side_length matrix ans assign the constellation position based on the gray code
    // everything is placed in a zig zag pattern
    // Y
    for (idx = 0; idx < side_length; ++idx)
    {
        // X
        for (jdx = 0; jdx < side_length; ++jdx)
        {
            // check row and perform zig-zag assignment
            index = ((idx & 0x01) == 1) ? (side_length * (idx + 1) - 1) - jdx : jdx + (side_length * idx);
 
            // assign to bit_mapper
            bit_mapper[gc[index]] = std::complex<float>(c_p[jdx], c_p[idx]);
        }
    }

    return bit_mapper;

}   // end of generate_qam_constellation

//-----------------------------------------------------------------------------
inline std::vector<std::complex<int16_t>> generate_qam(std::vector<int16_t>& data, uint64_t sample_rate, uint16_t num_bits, float symbol_length, float amplitude)
{
    uint32_t idx, jdx;
    uint16_t num = 0;

    std::vector<std::complex<float>> bit_mapper = generate_qam_constellation(num_bits);

    uint32_t samples_per_bit = floor(sample_rate * symbol_length + 0.5);

    // make sure that data has the right number of bits
    uint16_t n = mod(data.size(), num_bits);
    if (n != 0)
    {
        data.insert(data.end(), n, 0);
    }

    // get the number of bit groupings
    uint32_t num_bit_groups = floor(data.size() / num_bits);

    std::vector<complex<int16_t>> iq;

    for (idx = 0; idx < data.size(); idx += num_bits)
    {
        num = 0;
        for (jdx = 0; jdx < num_bits; ++jdx)
        {
            num += data[idx+jdx] << (num_bits - jdx);
        }

        iq.insert(iq.end(), samples_per_bit, amplitude* bit_mapper[num]);

    }

}   // end of generate_qam


//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx=0, jdx=0;

    typedef chrono::nanoseconds ns;
    auto start_time = chrono::high_resolution_clock::now();
    auto stop_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<ns>(stop_time - start_time);

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
        
        uint32_t img_h = 320;
        uint32_t img_w = 320;
        double fps = 30;
        int32_t four_cc = 0;

        std::vector<float> w = DSP::blackman_nuttall_window(601);
        
        num_loops = 100;

        const uint16_t num_bits = 6;
        uint32_t num = 1 << num_bits;
        std::vector<uint32_t> gc = gray_code(num_bits);
        std::bitset<num_bits> x;
        uint32_t side_length = 1<<(num_bits>>1);
        uint32_t index = 0;
        uint32_t shift = 0;

        std::vector<std::complex<float>> bit_mapper(1<< num_bits);
        float offset = (side_length <<1) - 1.0;

        // create the base locations for the constellation
        std::vector<float> t_x;
        double step = 2.0;
        int16_t start = side_length - offset;

        std::cout << "num_bits:    " << num_bits << std::endl;
        std::cout << "num:         " << num << std::endl;
        std::cout << "side_length: " << side_length << std::endl;
        std::cout << "offset:      " << offset << std::endl;
        std::cout << std::endl;

        for (idx = 0; idx < side_length; ++idx)
        {
            t_x.push_back(start);
            std::cout << start << "\t";

            start += step;

        }
        std::cout << std::endl << std::endl;

        uint32_t index2 = 0;

        for (idx = 0; idx < side_length; ++idx)
        {
            for (jdx = 0; jdx < side_length; ++jdx)
            {
                shift = (idx&0x01 == 1) ? (side_length -1)-jdx : jdx;
                x = (gc[index + shift]);
                bit_mapper[gc[index + shift]] = std::complex<float>(t_x[jdx], t_x[idx]);
                //bit_mapper[index2] = std::complex<float>(t_x[idx], t_x[jdx]);

                std::cout << index2 << "  " << gc[index + shift] << " [" << x << "] " << bit_mapper[gc[index + shift]] << "  ";
                ++index2;
            }
            index += side_length;
            std::cout << std::endl;

        }

        std::cout << std::endl;


        std::cout << "index\tgc\tbit\tbm[i]\tbm[gc[i]" <<std::endl;
        for (idx = 0; idx < gc.size(); ++idx)
        {
            x = gc[idx];
            std::cout << idx << "\t" << gc[idx] << "\t" << x << "\t" << bit_mapper[idx] << std::endl;
        }

        //index = 0;
        //for (idx = 0; idx < side_length; ++idx)
        //{
        //    for (jdx = 0; jdx < side_length; ++jdx)
        //    {
        //        x = gc[index];
        //        std::cout << index << " " << gc[index] << " " << x << " " << bit_mapper[gc[index]] << "  ";
        //        ++index;
        //    }
        //    
        //    std::cout << std::endl;

        //}

        std::cout << std::endl;


        std::vector<std::complex<float>> bit_mapper2 = generate_qam_constellation(num_bits);

        index = 0;
        for (idx = 0; idx < bit_mapper2.size(); ++idx)
        {
            x = gc[idx];
            std::cout << idx << "\t" << "\t" << x << "\t" << bit_mapper[idx] << "\t" << ((float)(side_length - 1.0)) * bit_mapper2[idx] << std::endl;
        }

        std::cout << std::endl;


        index = 0;
        for (idx = 0; idx < side_length; ++idx)
        {
            for (jdx = 0; jdx < side_length; ++jdx)
            {
                index = ((idx & 0x01) == 1) ? (side_length * (idx + 1) - 1) - jdx : jdx + (side_length * idx);
                x = gc[index];
                std::cout << index << " " << gc[index] << " " << x << " " << ((float)(side_length - 1.0)) * bit_mapper2[index] << "  \t";
                //++index;
            }
            std::cout << std::endl;
        }

        bp = 0;

        //uint32_t factor = 240;
        //uint64_t sample_rate = (1187.5*2.0) * factor;

        //std::cout << "sample_rate: " << sample_rate << std::endl;

        // create the data
        //rds_block_1 b1_0A(0x72C0);   // WLKI --> hex(11*676 + 10*26 + 8 + 21672) = hex(29376) = 72C0
        //rds_block_2 b2_0A(RDS_GROUP_TYPE::GT_0, RDS_VERSION::A, RDS_TP::TP_0, (5 << PTY_SHIFT), (RDS_TA::TA_0 | RDS_MS::MS_1 | RDS_DI3::DI3_0 | 0));
        //rds_block_3 b3_0A(224, 205);
        //rds_block_4 b4_0A('A', 'B');

        rds_params rp(0x72C0, RDS_VERSION::A, RDS_TP::TP_0, RDS_PTY::ROCK, RDS_TA::TA_0, RDS_MS::MS_1);

        std::string program_name = "TST_RDIO";
        std::string radio_text = "All Day All Night, We Know What You Need!";
        rds_generator rdg(rp);

        rdg.init_generator(program_name, radio_text);

        std::vector<complex<int16_t>> iq_data = rdg.generate_bit_stream();

        //int16_t previous_bit = 0;
        //data_bits = differential_encode(data_bits, previous_bit);

        //std::vector<float> data_bits_f = biphase_encode(data_bits);

        //std::cout << std::endl << "biphase out" << std::endl;
        //for (idx = 0; idx < data_bits_f.size(); ++idx)
        //{
        //    std::cout << (data_bits_f[idx]) << ", ";
        //}
        //std::cout << std::endl;

        //// upsample the data
        //std::vector<float> data_bits_u = upsample_data(data_bits_f, (factor>>1));
        ////std::vector<float> data_bits_u = upsample_data(data_bits_f, 1);

        //// filter the data
        //int64_t num_taps = factor + 1;       //data_bits_u .size();
        //float fc = 2200.0/(float)sample_rate;

        //std::vector<float> lpf = DSP::create_fir_filter<float>(num_taps, fc, &DSP::blackman_nuttall_window);

        //std::vector<float> rds;
        //apply_filter(data_bits_u, lpf, rds);

        //// create the pilot tone based on the data length and the rds rotation vector
        //uint64_t num_samples = rds.size();

        //float pilot_tone = 19000;
        //std::complex<float> j(0, 1);
        //const float math_2pi = 6.283185307179586476925286766559f;

        //std::vector<complex<float>> pt(num_samples, std::complex<float>(0,0));
        //std::vector<complex<float>> rds_rot(num_samples, std::complex<float>(0, 0));
        //std::vector<complex<int16_t>> iq_data(num_samples, std::complex<int16_t>(0, 0));

        //std::vector<float> audio_data(num_samples, 0);

        //// create audio tone
        //for (idx = 0; idx < num_samples; ++idx)
        //{
        //    audio_data[idx] = std::cos(M_2PI*(300 / (float)sample_rate)*idx);
        //}

        //std::vector<complex<float>> audio_fm = generate_fm(audio_data, sample_rate, 1, 0.8);

        //for (idx = 0; idx < num_samples; ++idx)
        //{
        //    //pt[idx] = std::complex<float>(200.0f, 0.0f) * std::exp(j * math_2pi * (float)((pilot_tone / (double)sample_rate) * idx));
        //    pt[idx] = std::complex<float>(400.0f * std::cos(math_2pi * (float)((pilot_tone / (double)sample_rate) * idx)), 0.0f);

        //    //rds_rot[idx] = std::complex<float>(160000.0f * rds[idx], 0.0f) * std::exp(j * math_2pi * (float)((3.0f * pilot_tone / (double)sample_rate) * idx));
        //    rds_rot[idx] = std::complex<float>(160000.0f * rds[idx] * std::cos(math_2pi * (float)((3.0f * pilot_tone / (double)sample_rate) * idx)), 0.0f) ;

        //    //iq_data[idx] = (pt[idx] + std::complex<float>(800.0f, 0.0f) * audio_fm[idx] + rds_rot[idx]);
        //    iq_data[idx] = (pt[idx] + rds_rot[idx]);
        //    //iq_data[idx] = rds_rot[idx];

        //}


        std:string savefile = "D:/Projects/data/RF/test_rds.sc16";
        //std:string savefile = "D:/data/RF/test_rds.sc16";

        save_complex_data(savefile, iq_data);


        /*
        std::vector<uint8_t> data;
        float amplitude = 2000;
        uint32_t sample_rate = 52000000;
        float half_bit_length = 0.00000025;
        uint32_t fc = 1200000;

        uint32_t num_bits = 208;
        uint32_t num_bursts = 16*16;

        std::vector<int32_t> channels = { -8000000, -7000000, -6000000, -5000000, -4000000, -3000000, -2000000, -1000000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000 };

        //std::vector<std::complex<int16_t>> iq_data;
        //std::vector<int16_t> iq_data;
        //int16_t *iq_data = NULL;

        // use these variables for the datatype > 0
        //typedef void (*init_)(long seed);
        //typedef void (*create_color_map_)(unsigned int h, unsigned int w, double scale, unsigned int octaves, double persistence, unsigned char* color, unsigned char* map);
        
        typedef void (*init_generator_)(float amplitude, uint32_t sample_rate, float half_bit_length, uint32_t filter_cutoff, uint32_t num_bits, int32_t * ch, uint32_t num_channels);
        typedef void (*generate_random_bursts_)(uint32_t num_bursts, uint32_t num_bits, int16_t** iq_ptr, uint32_t* data_size);
        
        
        HINSTANCE test_lib = NULL;

        init_generator_ init_generator;
        generate_random_bursts_ generate_random_bursts;

        std::string lib_filename;

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

        lib_filename = "D:/Projects/rpi_tester/x_compile/build/Release/test_gen.dll";

        test_lib = LoadLibrary(lib_filename.c_str());

        if (test_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        init_generator = (init_generator_)GetProcAddress(test_lib, "init_generator");
        generate_random_bursts = (generate_random_bursts_)GetProcAddress(test_lib, "generate_random_bursts");
#else
        lib_filename = "../../../rpi_tester/x_compile/build/libtest_gen.so";

        test_lib = dlopen(lib_filename.c_str(), RTLD_NOW);

        if (test_lib == NULL)
        {
            throw std::runtime_error("error loading library");
        }

        init_generator = (init_generator_)dlsym(test_lib, "init_generator");
        generate_random_bursts = (generate_random_bursts_)dlsym(test_lib, "generate_random_bursts");

#endif

        init_generator(amplitude, sample_rate, half_bit_length, fc, num_bits, channels.data(), (uint32_t)channels.size());

        uint32_t data_size = 0;
        generate_random_bursts(num_bursts, num_bits, &iq_data, &data_size);
        generate_random_bursts(num_bursts, num_bits, &iq_data, &data_size);

        //burst_generator bg(amplitude, sample_rate, half_bit_length, fc, num_bits, channels);

        //bg.generate_channel_rot(num_bits);

        double run_time_sum = 0.0;

        for(idx=0; idx<2; ++idx)
        {
            start_time = chrono::high_resolution_clock::now();

            //bg.generate_random_bursts(num_bursts*16, num_bits, iq_data);
            generate_random_bursts(num_bursts, num_bits, &iq_data, &data_size);

            stop_time = chrono::high_resolution_clock::now();

            const auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);

            std::cout << "elapsed_time: " << int_ms.count()/1e6 << std::endl;

            run_time_sum += int_ms.count() / 1.0e6;
        }

        //std::cout << "average elapsed_time: " << run_time_sum/100.0 << std::endl;
        //save_complex_data("D:/Projects/data/RF/test_oqpsk_burst.sc16", iq_data);
        

        std::string savefile;

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

        savefile = "D:/Projects/data/RF/test_oqpsk_burst.sc16";

#else
        savefile = "../../../data/RF/test_oqpsk_burst.sc16";

#endif
        save_complex_data(savefile, iq_data, data_size);
        */


        std::cout << "done saving data..." << std::endl;
        std::cin.ignore();


        bp = 10;
        
        //num_blocks = std::ceil(num_loops / (double)num_threads);

        /*
        'appsrc ! videoconvert' + \
    ' ! video/x-raw,format=I420' + \
    ' ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=' + str(fps * 2) + \
    ' ! video/x-h264,profile=baseline' + \
    ' ! rtspclientsink location=rtsp://localhost:8554/mystream'
        */
/*        
        //std::string cap_string = "rtsp://192.168.1.150:8554/temp";
        std::string cap_string = "rtsp://192.168.1.153:8554/camera-15";
        cv::VideoCapture cap(cap_string);
        cv::Mat input_frame;
        cap >> input_frame;

        img_h = input_frame.rows;
        img_w = input_frame.cols;

        std::string video_link;

//        video_link = "appsrc ! videoconvert ! video/x-raw, format=I420, format=BGR ! x264enc speed-preset=ultrafast key-int-max=60 ! video/x-h264, profile=baseline ! rtspclientsink protocols=tcp location=rtsp://192.168.1.150:8554/mystream";
        video_link = "appsrc ! videoconvert ! video/x-raw, format=I420 ! x264enc speed-preset=ultrafast key-int-max=60 ! video/x-h264, profile=baseline ! rtspclientsink protocols=tcp location=rtsp://192.168.1.153:8554/mystream";
        //video_link = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! video/x-h264,profile=high ! flvmux ! udpsink host=192.168.1.150/mystream port=8554";
        //video_link = "appsrc ! videoconvert ! udpsink host=192.168.1.150:8554/mystream ";

        cv::VideoWriter writer(video_link, cv::CAP_GSTREAMER, four_cc, fps, cv::Size(img_w, img_h), true);


        start_time = chrono::high_resolution_clock::now();
        num_threads = std::max(1U, std::thread::hardware_concurrency() - 1);

        cv::Mat previous_frame = cv::Mat::zeros(img_h, img_w, CV_32FC1);

        cv::Mat output_frame;
        cv::Mat temp_frame;
        cv::Mat mask_img;
        cv::Mat mask_invert = cv::Mat(img_h, img_w, CV_32FC3, cv::Scalar(1, 1, 1));

        //cv::cvtColor(output_frame, output_frame, cv::COLOR_BGR2RGB);

        cv::Mat block(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Rect block_rect(400, 150, block.cols, block.rows);

        cv::RNG rng;

        cv::namedWindow("test", cv::WINDOW_NORMAL);
        cv::namedWindow("input", cv::WINDOW_NORMAL);
        cv::namedWindow("diff_frame", cv::WINDOW_NORMAL);

        //cap >> input_frame;
        //cv::imwrite("baseline_img.png", input_frame);

        cv::Mat baseline_img = cv::imread("baseline_img.png");
        baseline_img.convertTo(baseline_img, CV_32FC1);

        cv::Rect overlay_rect;

        char key = 0;

        while (key != 'q')
        {
            start_time = chrono::high_resolution_clock::now();
            cap >> input_frame;

            if (input_frame.empty())
                continue;

            cv::imshow("input", input_frame);

            mask_img = cv::Mat::zeros(img_h, img_w, CV_32FC3);

            //rng.fill(block, cv::RNG::UNIFORM, 0, 255);
            //cv::rectangle(input_frame, cv::Rect(20, 130, 150, 120), cv::Scalar(0, 0, 0), -1);
            //block.copyTo(input_frame(block_rect));

            output_frame = input_frame.clone();
            cv::cvtColor(input_frame, input_frame, cv::COLOR_BGR2GRAY);
            input_frame.convertTo(input_frame, CV_32FC1);
            // blur the image using a sigma == 1.0 with border reflection
            cv::GaussianBlur(input_frame, input_frame, cv::Size(0, 0), 1.5, 1.5, cv::BORDER_REFLECT_101);
            cv::absdiff(input_frame, previous_frame, temp_frame);
            advanced_threshold(temp_frame, 6, 0.0f, 255.0f);

            cv::morphologyEx(temp_frame, temp_frame, cv::MORPH_DILATE, SE5_rect);
            cv::morphologyEx(temp_frame, temp_frame, cv::MORPH_CLOSE, SE5_rect);

            temp_frame.convertTo(temp_frame, CV_8UC1);
            cv::imshow("diff_frame", temp_frame);

            // find the contours of the remaining shapes
            cv::findContours(temp_frame, img_contours, img_hr, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (idx = 0; idx < img_contours.size(); ++idx)
            {
                get_rect(img_contours[idx], overlay_rect, img_w, img_h);

                cv::rectangle(mask_img, overlay_rect, cv::Scalar(1,1,1), -1);

                //baseline_img(overlay_rect).copyTo(output_frame(overlay_rect));

            }

            cv::GaussianBlur(mask_img, mask_img, cv::Size(0, 0), 2.0, 2.0, cv::BORDER_REFLECT_101);
            output_frame.convertTo(output_frame, CV_32FC3);
            output_frame = baseline_img.mul(mask_img) + output_frame.mul(mask_invert - mask_img);
            output_frame.convertTo(output_frame, CV_8UC3);

            //output_frame.convertTo(output_frame, CV_8UC3);
            //cv::cvtColor(output_frame, output_frame, cv::COLOR_GRAY2BGR);

            writer << output_frame;

            cv::imshow("test", output_frame);
            key = cv::waitKey(1);

            previous_frame = input_frame.clone();

            do
            {
                stop_time = chrono::high_resolution_clock::now();
                elapsed_time = chrono::duration_cast<ns>(stop_time - start_time);
            } while (elapsed_time.count() < 1 / fps);

        }




        //{
            //std::vector<std::thread> threads(num_threads);
            //std::mutex critical;
            //for (int t = 0; t < num_threads; t++)
            //{
            //    threads[t] = std::thread(std::bind([&](const int bi, const int ei, const int t)
            //    {

            //        // loop over all items
            //        for (int idx = bi; idx < ei; ++idx)
            //        {
            //            //for (idx = 0; idx < num_loops; ++idx)
            //            {
            //                Sleep(500);
            //                std::lock_guard<std::mutex> lock(critical);
            //                std::cout << "Index: " << idx << std::endl;
            //            }
            //        }
            //    }, t* num_loops / num_threads, (t + 1) == num_threads ? num_loops : (t + 1) * num_loops / num_threads, t));

            //}
            //std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });

        //}
        stop_time = chrono::high_resolution_clock::now();


        elapsed_time = chrono::duration_cast<ns>(stop_time - start_time);

        std::cout << "elapsed_time " << elapsed_time.count() << std::endl;

        //----------------------------------------------------------------------------------------
*/
        bp = 4;

    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }


    //cv::destroyAllWindows();

    std::cout << std::endl << "Press Enter to close" << std::endl;
    std::cin.ignore();

	return 0;

}	// end of main

