#ifndef _PSO_PARTICLE_H_
#define _PSO_PARTICLE_H_

#include <cstdint>
#include <ostream>
#include <istream>

#include <dlib/rand.h>
#include <dlib/matrix.h>
#include <dlib/serialize.h>


const uint64_t fc4_size = (1081 + 1) * 500;
const uint64_t fc3_size = (500 + 1) * 100;
const uint64_t fc2_size = (100 + 1) * 20;
const uint64_t fc1_size = (20 + 1) * 2;

// ----------------------------------------------------------------------------------------

class particle
{
private:

public:
    uint64_t number;
    uint64_t iteration;

    dlib::matrix<double, 1, fc1_size> x1;
    dlib::matrix<double, 1, fc2_size> x2;
    dlib::matrix<double, 1, fc3_size> x3;           // (200 + 1) * 50
    dlib::matrix<double, 1, fc4_size> x4;       //(1081 + 1) * 200


    particle() {}

    //particle(dlib::matrix<double> x_) : x(x_) {}
    particle(
        dlib::matrix<double> x1_,
        dlib::matrix<double> x2_,
        dlib::matrix<double> x3_,
        dlib::matrix<double> x4_
    ) : x1(x1_), x2(x2_), x3(x3_), x4(x4_) 
    {
        number = 0;
        iteration = 0;
    }

    dlib::matrix<double> get_x1() { return x1; }
    dlib::matrix<double> get_x2() { return x2; }
    dlib::matrix<double> get_x3() { return x3; }
    dlib::matrix<double> get_x4() { return x4; }

    void set_number(uint64_t n) { number = n; }

    uint64_t get_number(void) { return number; }

    // ----------------------------------------------------------------------------------------
    // This function is used to randomly initialize 
    void rand_init(dlib::rand& rnd, std::pair<particle, particle> limits)
    {
        long idx;

        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = rnd.get_double_in_range(limits.first.x1(0, idx), limits.second.x1(0, idx));
        }
        for (idx = 0; idx < x2.nc(); ++idx)
        {
            x2(0, idx) = rnd.get_double_in_range(limits.first.x2(0, idx), limits.second.x2(0, idx));
        }
        for (idx = 0; idx < x3.nc(); ++idx)
        {
            x3(0, idx) = rnd.get_double_in_range(limits.first.x3(0, idx), limits.second.x3(0, idx));
        }
        for (idx = 0; idx < x4.nc(); ++idx)
        {
            x4(0, idx) = rnd.get_double_in_range(limits.first.x4(0, idx), limits.second.x4(0, idx));
        }
    }

    // ----------------------------------------------------------------------------------------
    // This fucntion checks the particle value to ensure that the limits are not exceeded
    void limit_check(std::pair<particle, particle> limits)
    {
        long idx;

        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = std::max(std::min(limits.second.x1(0, idx), x1(0, idx)), limits.first.x1(0, idx));
        }
        for (idx = 0; idx < x2.nc(); ++idx)
        {
            x2(0, idx) = std::max(std::min(limits.second.x2(0, idx), x2(0, idx)), limits.first.x2(0, idx));
        }
        for (idx = 0; idx < x3.nc(); ++idx)
        {
            x3(0, idx) = std::max(std::min(limits.second.x3(0, idx), x3(0, idx)), limits.first.x3(0, idx));
        }
        for (idx = 0; idx < x4.nc(); ++idx)
        {
            x4(0, idx) = std::max(std::min(limits.second.x4(0, idx), x4(0, idx)), limits.first.x4(0, idx));
        }
    }

    // ----------------------------------------------------------------------------------------
    static particle get_rand_particle(dlib::rand& rnd)
    {
        long idx;
        particle p;

        for (idx = 0; idx < p.x1.nc(); ++idx)
        {
            p.x1(0, idx) = rnd.get_random_double();
        }
        for (idx = 0; idx < p.x2.nc(); ++idx)
        {
            p.x2(0, idx) = rnd.get_random_double();
        }
        for (idx = 0; idx < p.x3.nc(); ++idx)
        {
            p.x3(0, idx) = rnd.get_random_double();
        }
        for (idx = 0; idx < p.x4.nc(); ++idx)
        {
            p.x4(0, idx) = rnd.get_random_double();
        }

        return p;
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator+(const particle& p1, const particle& p2)
    {
        return particle(p1.x1 + p2.x1, p1.x2 + p2.x2, p1.x3 + p2.x3, p1.x4 + p2.x4);
        //return particle(p1.x + p2.x);
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator-(const particle& p1, const particle& p2)
    {
        return particle(p1.x1 - p2.x1, p1.x2 - p2.x2, p1.x3 - p2.x3, p1.x4 - p2.x4);
        //return particle(p1.x - p2.x);
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator*(const particle& p1, const particle& p2)
    {
        return particle(dlib::pointwise_multiply(p1.x1, p2.x1), dlib::pointwise_multiply(p1.x2, p2.x2),
            dlib::pointwise_multiply(p1.x3, p2.x3), dlib::pointwise_multiply(p1.x4, p2.x4));
        //return particle(dlib::pointwise_multiply(p1.x, p2.x));
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator*(const particle& p1, double& v)
    {
        return particle(v * p1.x1, v * p1.x2, v * p1.x3, v * p1.x4);
        //return particle(v * p1.x);
    }

    // ----------------------------------------------------------------------------------------
    inline friend const particle operator*(double& v, const particle& p1)
    {
        return particle(v * p1.x1, v * p1.x2, v * p1.x3, v * p1.x4);
        //return particle(v * p1.x);
    }

    // ----------------------------------------------------------------------------------------
    friend void serialize(const particle& item, std::ostream& out)
    {
        dlib::serialize("base_particle", out);
        dlib::serialize(item.number, out);
        dlib::serialize(item.iteration, out);
        dlib::serialize(item.x1, out);
        dlib::serialize(item.x2, out);
        dlib::serialize(item.x3, out);
        dlib::serialize(item.x4, out);
    }

    // ----------------------------------------------------------------------------------------
    friend void deserialize(particle& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "base_particle")
            throw dlib::serialization_error("Unexpected version found: " + version + " while deserializing particle.");
        dlib::deserialize(item.number, in);
        dlib::deserialize(item.iteration, in);
        dlib::deserialize(item.x1, in);
        dlib::deserialize(item.x2, in);
        dlib::deserialize(item.x3, in);
        dlib::deserialize(item.x4, in);
    }

    // ----------------------------------------------------------------------------------------
    inline friend std::ostream& operator<< (std::ostream& out, const particle& item)
    {
        out << "x=" << dlib::csv << item.x1;
        return out;
    }
};


#endif  // _PSO_PARTICLE_H_
