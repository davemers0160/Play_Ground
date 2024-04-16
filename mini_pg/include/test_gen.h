#ifndef TEST_GEN_H_
#define TEST_GEN_H_

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
// need for VS for pi and other math constatnts
#define _USE_MATH_DEFINES

#elif defined(__linux__)

#endif

// C/C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <complex>
#include <random>
//#include <algorithm>
//#include <thread>
//#include <mutex>

#include "dsp/dsp_windows.h"


//-----------------------------------------------------------------------------
inline std::vector<std::complex<float>> generate_oqpsk(std::vector<uint8_t> data, double amplitude, uint32_t sample_rate, float half_symbol_length)
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

    std::vector<std::complex<float>> iq_data(num_bit_pairs * samples_per_symbol + samples_per_bit);

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
        iq_data[idx] = (std::complex<float>(I[idx], Q[idx]));
    }

    return iq_data;
}   // end of generate_oqpsk


//-----------------------------------------------------------------------------
class burst_generator
{
//-----------------------------------------------------------------------------
public:

	burst_generator()
	{
		amplitude = 2000;
		sample_rate = 50000000;
        half_symbol_length = 0.0000001;

		data.clear();
        cpf.resize(n_taps);

        generator = std::default_random_engine(time(0));
        bits_gen = std::uniform_int_distribution<int32_t>(0, 1);

        channel_gen = std::uniform_int_distribution<int32_t>(0, channels.size()-1);

        create_filter();

        uint32_t num_samples = floor(sample_rate * half_symbol_length);
        generator_channel_rot(num_samples);
	}

	burst_generator(double a, uint32_t fs, float hst) : amplitude(a), sample_rate(fs), half_symbol_length(hst)
	{
		data.clear();
        cpf.resize(n_taps);

        generator = std::default_random_engine(time(0));
        bits_gen = std::uniform_int_distribution<int32_t>(0, 1);

        channel_gen = std::uniform_int_distribution<int32_t>(0, channels.size() - 1);

        create_filter();

        uint32_t num_samples = floor(sample_rate * half_symbol_length);
        generator_channel_rot(num_samples);
	}

	//-----------------------------------------------------------------------------
    void generate_burst(uint32_t num_bursts, uint32_t num_bits, std::vector<std::complex<int16_t>> &IQ)
    {
        uint32_t idx, jdx;
        std::vector<uint8_t> data(num_bits);

        uint32_t ch, ch_rnd;

        std::vector<std::complex<float>> x1;
        std::vector<std::complex<int16_t>> x2;

        IQ.clear();

        for (jdx = 0; jdx < num_bursts; ++jdx)
        {

            ch_rnd = channel_gen(generator);
            //ch = channels[ch_rnd];

            // create the random bit sequence
            for (idx = 0; idx < num_bits; ++idx)
                data[idx] = bits_gen(generator);

            // generate the IQ data
            std::vector<std::complex<float>> iq_data = generate_oqpsk(data, amplitude, sample_rate, half_symbol_length);

            // filter the IQ samples
            apply_filter(iq_data, x1);

            // rotate the samples
            apply_rotation(x1, ch_rot[ch_rnd], x2);
            //apply_rotation(iq_data, ch_rot[ch_rnd], x2);

            //apply_filter_rotation(iq_data, ch_rot[ch_rnd], x2);

            // append the samples
            IQ.insert(IQ.end(), x2.begin(), x2.end());

        }
    }   // end of generate_burst

    //-----------------------------------------------------------------------------
    void generator_channel_rot(uint32_t num_bits)
    {
        uint32_t idx, jdx;

        double ch;

        if (num_bits % 2 == 1)
            ++num_bits;

        uint32_t num_bit_pairs = num_bits >> 1;
        uint32_t samples_per_bit = floor(sample_rate * half_symbol_length);

        uint32_t num_samples = num_bit_pairs * (samples_per_bit << 1) + samples_per_bit;

        ch_rot.clear();
        ch_rot.resize(channels.size());

        for (idx = 0; idx < channels.size(); ++idx)
        {
            ch_rot[idx].resize(num_samples);

            ch = M_2PI * (channels[idx] / (double)sample_rate);

            for (jdx = 0; jdx < num_samples; ++jdx)
            {
                ch_rot[idx][jdx] = std::exp(1i * (ch * jdx));
            }
        }

    }   // end of generator_channel_rot

//-----------------------------------------------------------------------------
private:
	//uint8_t burst_type;
	std::vector<uint8_t> data;
	float amplitude = 2000;
	uint32_t sample_rate = 50000000;
	float half_symbol_length = 0.0000001;

    // window/filter size
    const int32_t n_taps = 31;
    std::vector<std::complex<float>> cpf;

    std::default_random_engine generator;
    std::uniform_int_distribution<int32_t> bits_gen;
    std::uniform_int_distribution<int32_t> channel_gen;

    std::vector<std::vector<std::complex<float>>> ch_rot;

    const float M_2PI = 6.283185307179586476925286766559;

    //uint32_t num_bursts;

    std::vector<int32_t> channels = {-8000000, -7000000, -6000000, -5000000, -4000000, -3000000, -2000000, -1000000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000 };

    //-----------------------------------------------------------------------------
    void create_filter()
    {
        uint32_t idx, jdx;

        // filter cutoff frequency
        float fc_m = 4.0e6 / (float)sample_rate;
        float fc_h = 29.0e6 / (float)sample_rate;

        // create the low pass filter
        std::vector<float> lpf = DSP::create_fir_filter<float>(n_taps, fc_m, &DSP::nuttall_window);
        std::vector<float> apf = DSP::create_fir_filter<float>(n_taps, 1.0, &DSP::nuttall_window);
        std::vector<float> hpf = DSP::create_fir_filter<float>(n_taps, fc_h, &DSP::nuttall_window);

        //cpf = (lpf + 0.3 * (apf - hpf)) / 2.0;

        for (idx = 0; idx < n_taps; ++idx)
        {
            cpf[idx] = std::complex <float>(0.5 * (lpf[idx] + (0.3 * (apf[idx] - hpf[idx]))), 0.0f);
        }

    }   // end of create_filter

    //-----------------------------------------------------------------------------
    void apply_filter(std::vector<std::complex<float>> &iq_data, std::vector<std::complex<float>> &x1)
    {
        uint32_t idx, jdx;
        uint32_t offset = n_taps >> 1;

        std::complex<float> accum;

        x1.clear();
        x1.resize(iq_data.size(), std::complex<float>(0, 0));
        int32_t temp = 0;

        for (idx = offset; idx < (iq_data.size() - offset); ++idx)
        {
            accum = 0.0;
            temp = idx - offset;

            for (jdx = 0; jdx < n_taps; ++jdx)
            {
                //std::complex<double> t1 = std::complex<double>(lpf[jdx], 0);
                //std::complex<double> t2 = iq_data[idx + jdx - offset];
                accum += iq_data[temp + jdx] * cpf[jdx];
            }

            x1[temp] = accum;
        }

    }   // end of apply_filter

    //-----------------------------------------------------------------------------
    void apply_filter_rotation(std::vector<std::complex<float>>& iq_data, std::vector<std::complex<float>>& f_rot, std::vector<std::complex<int16_t>>& x1)
    {
        uint32_t idx, jdx;
        uint32_t offset = n_taps >> 1;

        std::complex<float> accum;

        x1.clear();
        x1.resize(iq_data.size(), std::complex<float>(0, 0));

        int32_t temp = 0;

        for (idx = offset; idx < (iq_data.size() - offset); ++idx)
        {
            accum = 0.0;
            temp = idx - offset;
            for (jdx = 0; jdx < n_taps; ++jdx)
            {
                //std::complex<double> t1 = std::complex<double>(lpf[jdx], 0);
                //std::complex<double> t2 = iq_data[idx + jdx - offset];
                accum += iq_data[temp + jdx] * cpf[jdx];
            }

            x1[temp] = std::complex<int16_t>(f_rot[temp] * accum);

        }

    }   // end of apply_filter

    //-----------------------------------------------------------------------------
    void apply_rotation(std::vector<std::complex<float>>& src, std::vector<std::complex<float>> &f_rot, std::vector<std::complex<int16_t>>& dst)
    {
        uint32_t idx;

        // apply frequency rotation
        dst.clear();
        dst.resize(src.size(), std::complex<int16_t>(0, 0));

        for (idx = 0; idx < src.size(); ++idx)
        {
            dst[idx] = std::complex<int16_t>(f_rot[idx] * src[idx]);
        }
    }

};	// endof class


#endif	// TEST_GEN_H_
