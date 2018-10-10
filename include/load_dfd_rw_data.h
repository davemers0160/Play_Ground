#ifndef LOAD_DFD_RW_DATA_H
#define LOAD_DFD_RW_DATA_H

// This loading function assumes that the ground truth image size and the input image sizes do not have to be the same dimensions

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <string>

// Custom Includes
#include "rgb2gray.h"
#include "ycrcb_pixel.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>


extern const uint32_t img_depth;
extern const uint32_t secondary;

void load_dfd_rw_data(
    const std::vector<std::vector<std::string>> training_file, 
    const std::string data_directory,
    std::vector<std::array<dlib::matrix<uint16_t>, img_depth>> &t_data,
    std::vector<dlib::matrix<uint16_t>> &gt,
    std::vector<std::pair<std::string, std::string>> &image_files
)
{

    int idx;
    
    std::string imageLocation;
    std::string FocusFile;
    std::string DefocusFile;
    std::string GroundTruthFile;
    
    // clear out the container for the focus and defocus filenames
    image_files.clear();

    dlib::matrix<int16_t> lap_kernel(3, 3);
    lap_kernel = 1, 1, 1, 1, -8, 1, 1, 1, 1;


    for (idx = 0; idx < training_file.size(); idx++)
    {

        // read in the ground truth image
        GroundTruthFile = data_directory + training_file[idx][2];

        FocusFile = data_directory + training_file[idx][0];
        DefocusFile = data_directory + training_file[idx][1];

        image_files.push_back(std::make_pair(FocusFile, DefocusFile));
        
        // load in the data files
        std::array<dlib::matrix<uint16_t>, img_depth> t;
        dlib::matrix<uint16_t> tf, td;
        dlib::matrix<dlib::rgb_pixel> f, f_tmp, d, d_tmp;
        dlib::matrix<int16_t> horz_gradient, vert_gradient;

        dlib::matrix<uint16_t> g, g_tmp;

        // load the images into an rgb_pixel format
        // @mem((f.data).data, UINT8, 3, f.nc(), f.nr(), f.nc()*3)
        std::string focus_ext = FocusFile.substr(FocusFile.length() - 3, FocusFile.length() - 1);
        std::string defocus_ext = DefocusFile.substr(DefocusFile.length() - 3, DefocusFile.length() - 1);
        std::string gt_ext = GroundTruthFile.substr(GroundTruthFile.length() - 3, GroundTruthFile.length() - 1);


        // dlib::load_image(f_tmp, FocusFile);
        // dlib::load_image(d_tmp, DefocusFile);
        // dlib::load_image(g_tmp, GroundTruthFile);
        dlib::load_image(f, FocusFile);
        dlib::load_image(d, DefocusFile);
        dlib::load_image(g, GroundTruthFile);
        
        // crop the images to the right network size
        // get image size
        // long rows = f_tmp.nr();
        // long cols = f_tmp.nc();

        // crop image based on limitations of the network and scaling behavior
        // int row_rem = (rows) % 16;
        // if (row_rem != 0)
        // {
            // rows -= row_rem;
        // }

        // int col_rem = (cols) % 16;
        // if (col_rem != 0)
        // {
            // cols -= col_rem;
        // }

        // f.set_size(rows, cols);
        // d.set_size(rows, cols);
        // g.set_size(rows, cols);
        // dlib::set_subm(f, 0, 0, rows, cols) = dlib::subm(f_tmp, 0, 0, rows, cols);
        // dlib::set_subm(d, 0, 0, rows, cols) = dlib::subm(d_tmp, 0, 0, rows, cols);
        // dlib::set_subm(g, 0, 0, rows, cols) = dlib::subm(g_tmp, 0, 0, rows, cols);
        
        switch (img_depth)
        {
            case 1:
                bgr2gray(f, tf);
                bgr2gray(d, td);
                
                t[0] = (td+256)-tf;
                
                break;
            
            case 2:
                bgr2gray(f, t[0]);
                bgr2gray(d, t[1]);
                break;
                
            case 3:

                bgr2gray(f, t[0]);
                bgr2gray(d, t[1]);

                switch(secondary)
                {
                   case 1:
                        t[2] = dlib::matrix_cast<uint16_t>(dlib::abs(dlib::matrix_cast<float>(t[1])- dlib::matrix_cast<float>(t[0])));
                        break;

                    case 2:
                        dlib::sobel_edge_detector(t[0], horz_gradient, vert_gradient);                               
                        dlib::assign_image(t[2], dlib::abs(horz_gradient)+dlib::abs(vert_gradient));   
                        break;

                    case 3:
                        dlib::spatially_filter_image(t[0], t[2], lap_kernel, 1, true, false);
                        break;

                    default:
                        break;
                }
                break;
                
            case 6:
                // get the images size and resize the t array
                for (int m = 0; m < 6; ++m)
                {
                    t[m].set_size(f.nr(), f.nc());
                }

                switch (secondary)
                {
                    // RGB version with each color channel going into it's own layer
                    case 1:
                        // loop through the images and assign each color channel to one 
                        // of the array's of t
                        for (long r = 0; r < f.nr(); ++r)
                        {
                            for (long c = 0; c < f.nc(); ++c)
                            {
                                dlib::rgb_pixel p;
                                dlib::assign_pixel(p, f(r, c));
                                dlib::assign_pixel(t[0](r, c), p.red);
                                dlib::assign_pixel(t[1](r, c), p.green);
                                dlib::assign_pixel(t[2](r, c), p.blue);
                                dlib::assign_pixel(p, d(r, c));
                                dlib::assign_pixel(t[3](r, c), p.red);
                                dlib::assign_pixel(t[4](r, c), p.green);
                                dlib::assign_pixel(t[5](r, c), p.blue);
                            }
                        }
                        break;

                    // YCrCb version with each color channle going into it's own layer
                    case 2:
                        for (long r = 0; r < f.nr(); ++r)
                        {
                            for (long c = 0; c < f.nc(); ++c)
                            {
                                dlib::ycrcb_pixel p;
                                dlib::assign_pixel(p, f(r, c));
                                dlib::assign_pixel(t[0](r, c), p.y);
                                dlib::assign_pixel(t[1](r, c), p.cr);
                                dlib::assign_pixel(t[2](r, c), p.cb);
                                dlib::assign_pixel(p, d(r, c));
                                dlib::assign_pixel(t[3](r, c), p.y);
                                dlib::assign_pixel(t[4](r, c), p.cr);
                                dlib::assign_pixel(t[5](r, c), p.cb);
                            }
                        }
                        
                        break;

                    case 3:
                        for (long r = 0; r < f.nr(); ++r)
                        {
                            for (long c = 0; c < f.nc(); ++c)
                            {
                                dlib::lab_pixel p;
                                dlib::assign_pixel(p, f(r, c));
                                dlib::assign_pixel(t[0](r, c), p.l);
                                dlib::assign_pixel(t[1](r, c), p.a);
                                dlib::assign_pixel(t[2](r, c), p.b);
                                dlib::assign_pixel(p, d(r, c));
                                dlib::assign_pixel(t[3](r, c), p.l);
                                dlib::assign_pixel(t[4](r, c), p.a);
                                dlib::assign_pixel(t[5](r, c), p.b);
                            }
                        }
                        break;

                    default:
                        break;
                }
                break;
        }
        
        // @mem((t[0].data).data, UINT16, 1, t[0].nc(), t[0].nr(), t[0].nc()*2)
        t_data.push_back(t);
        gt.push_back(g);

    }   // end of the read in data loop

}   // end of load_dfd_rw_data


#endif  // LOAD_DFD_RW_DATA_H