#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h>
#include <algorithm>
#include <time.h>
#include <getopt.h>
#include <openpose/experimental/tracking/lkpyramidal.hpp>
//#define DEBUG

// #ifdef DEBUG
// /* When debugging is enabled, these form aliases to useful functions */
// #define dbg_printf(...) printf(__VA_ARGS__); 
// #else
// /* When debugging is disabled, no code gets generated for these */
// #define dbg_printf(...)
// #endif

#define SUCCESS 0
#define INVALID_PATCH_SIZE 1
#define OUT_OF_FRAME 2
#define ZERO_DENOMINATOR 3

using namespace cv;
using namespace std;



namespace op
{
    char compute_lk(vector<float> &ix, vector<float> &iy,
                    vector<float> &it, pair<float,float> &delta)
    {
        float sum_xx = 0.0, sum_yy = 0.0, sum_xt = 0.0,
                            sum_yt = 0.0, sum_xy = 0.0;
        float num_u,num_v, den, u, v;

        /* Calculate sums */
        for (auto i = 0u; i < ix.size(); i++)
        {
            sum_xx += ix[i] * ix[i];
            sum_yy += iy[i] * iy[i];
            sum_xy += ix[i] * iy[i];
            sum_xt += ix[i] * it[i];
            sum_yt += iy[i] * it[i];
        }

        /* Get numerator and denominator of u and v */
        den = (sum_xx*sum_yy) - (sum_xy * sum_xy);

        if (den == 0.0) return ZERO_DENOMINATOR;

        num_u = (-1.0 * sum_yy * sum_xt) + (sum_xy * sum_yt);
        num_v = (-1.0 * sum_xx * sum_yt) + (sum_xt * sum_xy);

        u = num_u / den;
        v = num_v / den;
        delta.first = u;
        delta.second = v;

        return SUCCESS;
    }

    void get_vectors(vector< vector<float> > &patch, 
                     vector< vector<float> > &patch_it,
                     int patch_size, vector<float> &ix, vector<float> &iy, 
                     vector<float> &it)
    {
        for (int i = 1; i <= patch_size; i++)
            for (int j = 1; j <= patch_size; j++)
            {
                ix.push_back((patch[i][j+1] - patch[i][j-1])/2.0);
                iy.push_back((patch[i+1][j] - patch[i-1][j])/2.0);
            }

        for (auto i = 0; i < patch_size; i++)
            for (int j = 0; j < patch_size; j++)
                it.push_back(patch_it[i][j]);

    }

    char extract_patch(int x, int y, int patch_size,
                       Mat &image, vector< vector<float> > &patch)
    {
        int radix = patch_size / 2;

        if ( ((x - radix) < 0) ||
             ((x + radix) >= image.cols) ||
             ((y - radix) < 0) ||
             ((y + radix) >= image.rows))
            return OUT_OF_FRAME;

        for (int i = -radix; i <= radix; i++)
            for (int j = -radix; j <= radix; j++)
                patch[i+radix][j+radix] = image.at<float>(y+i,x+j);

        return SUCCESS;

    }

    char extract_it_patch(int x_I, int y_I, int x_J, int y_J, Mat &I, Mat &J, 
                          int patch_size, vector< vector<float> > &patch)
    {

        int radix = patch_size / 2;

        if (((x_I - radix) < 0) ||
             ((x_I + radix) >= I.cols) ||
             ((y_I - radix) < 0) ||
             ((y_I + radix) >= I.rows))
            return OUT_OF_FRAME;

        if (((x_J - radix) < 0) ||
             ((x_J + radix) >= J.cols) ||
             ((y_J - radix) < 0) ||
             ((y_J + radix) >= J.rows))
            return OUT_OF_FRAME;

        for (int i = -radix; i <= radix; i++)
            for (int j = -radix; j <= radix; j++)
                patch[i+radix][j+radix] = J.at<float>(y_J+i,x_J+j) - 
                                          I.at<float>(y_I+i,x_I+j);

        return SUCCESS;

    }
    /* Given an OpenCV image 'img', build a gaussian pyramid of size 'levels' */
    void build_gaussian_pyramid(Mat &img, int levels, vector<Mat> &pyramid)
    {
        pyramid.clear();

        pyramid.push_back(img);

        for(int i = 0; i < levels - 1; i++)
        {
            Mat tmp;
            pyrDown(pyramid[pyramid.size() - 1], tmp);
            pyramid.push_back(tmp);
        }
    }



    Point2f pyramid_iteration(Point2f ipoint, Point2f jpoint, Mat &I, Mat &J,
                              char &status, int patch_size = 5)
    {

        Point2f result;

       /* Extract a patch around the image */
        vector< vector<float> > patch(patch_size + 2,
                                    vector<float>(patch_size + 2));
        vector< vector<float> > patch_it(patch_size,
                                    vector<float>(patch_size));

        status = extract_patch((int)ipoint.x,(int)ipoint.y,
                               patch_size + 2, I, patch);
    //    if (status)
    //        return result;

        status = extract_it_patch(ipoint.x, ipoint.y, jpoint.x, jpoint.y, I, J,
                                  patch_size, patch_it);
     
    //    if (status)
    //        return result;                         

        /* Get the Ix, Iy and It vectors */
        vector<float> ix, iy, it;
        get_vectors(patch, patch_it, patch_size, ix, iy, it);

        /* Calculate optical flow */
        pair<float,float> delta;
        status = compute_lk(ix, iy, it, delta);
        
    //    if (status)
    //        return result;
        
        result.x = jpoint.x + delta.first;
        result.y = jpoint.y + delta.second;

        return result; 
    }  

    void reescale_cords(vector<Point2f> &coords, float scale)
    {
        for (auto i = 0u; i < coords.size(); i++)
        {
            coords[i].x =  scale * coords[i].x;
            coords[i].y =  scale * coords[i].y;
        }
    }


    void reescale_cord(Point2f &coord, float scale)
    {
        coord.x =  scale * coord.x;
        coord.y =  scale * coord.y;
    }

    void runLKPyramidal(std::vector<cv::Point2f> &coord_I,
                         std::vector<cv::Point2f> &coord_J,
                         cv::Mat &prev,
                         cv::Mat &next,
                         std::vector<char> &status,
                         int levels,
                         int patch_size)
    {
        /* Empty coordinates */
        if (coord_I.size() == 0)
            return;

        vector<Point2f> I;
        I.assign(coord_I.begin(), coord_I.end());

        reescale_cords(I,1.0/(float)(1<<(levels-1)));
        
        coord_J.clear();
        coord_J.assign(I.begin(), I.end());

        vector<Mat> I_pyr;
        vector<Mat> J_pyr;

        build_gaussian_pyramid(prev, levels, I_pyr);
        build_gaussian_pyramid(next, levels, J_pyr);

        
        /* Process all pixel requests */
        for (auto i = 0u; i < coord_I.size(); i++)
        {
            for (int l = levels - 1; l >= 0; l--)
            {
                 char status_point = 0;
                 Point2f result;

                 result = pyramid_iteration(I[i], coord_J[i],I_pyr[l], J_pyr[l],
                                  status_point, patch_size);
                 if (status_point) {status[i] = status_point;}

                 coord_J[i] = result;
                
                 if (l == 0) break;

                 reescale_cord(I[i],2.0);
                 reescale_cord(coord_J[i],2.0);
            }
        }
    }
}
