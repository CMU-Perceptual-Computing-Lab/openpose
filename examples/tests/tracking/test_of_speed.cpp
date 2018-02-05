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
#include "CycleTimer.h"
#include <time.h>
#include <getopt.h>
#include <openpose/experimental/tracking/pyramidalLK.hpp>

#define SUCCESS 0
#define INVALID_PATCH_SIZE 1
#define OUT_OF_FRAME 2
#define ZERO_DENOMINATOR 3

using namespace cv;
using namespace std;
using namespace op;

string test_dir = "./examples/tests/tracking/data_speed_test/";
string format_img = ".jpg";
/* CPU/GPU mode */
bool use_gpu = false;
double total_time = 0.0;


void benchmark_lk(Mat image1_f, Mat image2_f, int npoints, int iter_no)
{

    srand (time(NULL));
    vector<char> custom_status(npoints,0);
    vector<float> err;
    vector<Point2f> tracking_points[2];
    int w = image1_f.cols;
    int h = image1_f.rows;
    double start = 0.0;
    double end = 0.0;
    /* Randomly get 'n' rando points in the width and height of the image
       range
     */
    for (int i = 0; i < npoints; i++)
    {
        int x = rand() % w;
        int y = rand() % h;    
        tracking_points[0].push_back(Point2d(x, y)); 
    }

    start = CycleTimer::currentSeconds();

    /* Benchmark selected version */
    if (use_gpu)
    {
        pyramidalLKGpu(tracking_points[0], tracking_points[1],
                       custom_status, image1_f, image2_f, 5, 21);
    }
    else
    {
        std::vector<cv::Mat> pyramidImagesPrevious;
        std::vector<cv::Mat> pyramidImagesCurrent;
        pyramidalLKCpu(tracking_points[0], tracking_points[1],
                       pyramidImagesPrevious, pyramidImagesCurrent,
                       custom_status, image1_f, image2_f, 5, 21); 
    }

    

    end = CycleTimer::currentSeconds();
    double total = end - start;
    
    if (iter_no > 1) total_time += total;


}

int main(int argc, char ** argv) 
{

    /* Read arguments */
    int option = 0, no_iters = 0, no_points = 0;
    /* Get input arguments */ 
    while ((option = getopt(argc, argv,"i:n:grd:")) != -1) 
    {
        switch (option) {
            case 'i' : no_iters = atoi(optarg);
                break;
            case 'n' : no_points = atoi(optarg);
                break;
            case 'g' : use_gpu = true; 
                break;
            case 'd' : test_dir = optarg;
                break;
            default:  
                exit(EXIT_FAILURE);
        }
    }

    string s_image1 = test_dir + "001" + format_img;
    string s_image2 = test_dir + "002" + format_img;

    Mat f_image1 = cv::imread(s_image1, CV_LOAD_IMAGE_GRAYSCALE);
    Mat f_image2 = cv::imread(s_image2, CV_LOAD_IMAGE_GRAYSCALE);
    
    /* Convert images to float */
    Mat image1_f, image2_f;
    f_image1.convertTo(image1_f, CV_32F);
    f_image2.convertTo(image2_f, CV_32F);

    for (int k = 0; k < no_iters; k++)
        benchmark_lk(image1_f, image2_f, no_points, k);

    /* Calculate average, and skip first two due to initialization
       overhead (GPU)
    */
    double average = total_time / (no_iters-2);

    cout<<"Finished running Optical Flow. Average in ms is "<<average*1e3<<endl;
    cout<<"Average per keypoint in ms is: "<< average / no_points<<endl; 

    return 0;
}
