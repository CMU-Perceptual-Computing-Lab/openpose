#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <algorithm>
#include <getopt.h>
#include <openpose/experimental/tracking/pyramidalLK.hpp>

using namespace cv;
using namespace std;
using namespace op;

/* Keypoints per person */
int keypoints = 18;
string format_img = ".jpg";
/* Interpolation pixel maximum distance */
const int op_pixel_threshold = 10; 

void drawKeyPoints(Mat image,vector<int> x, vector<int> y, 
                   std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for (auto i= 0u; i < x.size(); i++)
    {
        if (!x[i] && !y[i]) continue;

        cv::Point center = cv::Point(x[i], y[i]);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    imwrite(output_file, target);
}


void draw_all(Mat image, vector<int> x, vector<int> y, vector<Point2f> &of, 
              vector<Point2f> &custom, std::string output_file)
{

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for (auto i = 0u; i<x.size(); i++)
    {
        if (!x[i] && !y[i]) continue;

        cv::Point center = cv::Point(x[i],y[i]);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    for (auto i = 0u; i < of.size(); i++)
        cv::circle(target,of[i],3,Scalar(0,255,0),1);

    for (auto i = 0u; i < custom.size(); i++)
        cv::circle(target,custom[i],3,Scalar(0,0,255),1);

    imwrite(output_file, target);
}


void points2crd(vector<int> x, vector<int> y, vector<Point2f> &output_crd){

    for (auto i = 0u; i < x.size(); i++)
    {
        if (!x[i] && !y[i]) continue;

        cv::Point p = cv::Point2d((float)x[i], (float)y[i]);
        output_crd.push_back(p);
    }

}

void crd2points(vector<float> &x, vector<float> &y, vector<Point2f> &input_crd)
{
    x.clear();
    y.clear();

    for (auto i = 0u; i < input_crd.size(); i++)
    {
        x.push_back(input_crd[i].x);
        y.push_back(input_crd[i].y);
    }
}

pair<int,int> find_closest(vector<int> &x_of, vector<int> &y_of, int x, int y)
{
    float min_dist = (float) (INT_MAX);
    int best_x = -1, best_y = -1;

    for (auto i = 0u; i < x_of.size(); i++)
    {
        float dist = (float) ((x-x_of[i])*(x-x_of[i]) + 
                             (y-y_of[i])*(y-y_of[i]));
        if (dist < min_dist)
        {        
            min_dist = dist;
            best_x = x_of[i];
            best_y = y_of[i];
        }    
    }
    
    if (min_dist >= op_pixel_threshold)
        return make_pair(-1,-1);
    
    return make_pair(best_x,best_y);
}


void interpolate_next(vector<int> &x_prev, vector<int> &y_prev, 
                      vector<int> &x_of, vector<int> &y_of)
{

    for (auto i = 0u; i < x_prev.size(); i++)
    {
        if (x_prev[i] == -1)
            continue;

        pair<int,int> new_cord = find_closest(x_of,y_of,x_prev[i],y_prev[i]);
        x_prev[i] = new_cord.first;
        y_prev[i] = new_cord.second;
    }
}

void get_statistics(vector<int> &x_itp, vector<int> &y_itp,
                    vector<float> &x_of, vector<float> &y_of,
                    vector<float> &maxim, vector<float> &mean,
                    vector<float> &minim, vector<float> &median)
{

    vector<float> distances;
    float dist = 0.0, total = 0.0;
    int n = x_itp.size();

    for (auto i = 0u; i < x_itp.size(); i++)
    {
        if (x_itp[i] == -1)
            continue;
        
        dist = (float) (((float)x_itp[i]-x_of[i])*((float)x_itp[i]-x_of[i]) + 
                ((float)y_itp[i]-y_of[i])*((float)y_itp[i]-y_of[i]));        
        dist = sqrt(dist);
        distances.push_back(dist);         
    }

    if (!distances.size()) return;

    std::sort(distances.begin(),distances.end());
    int nk = distances.size(); 
    for (auto k = 0; k < nk; k++) total += distances[k];

    maxim.push_back(distances.back());
    minim.push_back(distances[0]);
    mean.push_back(total/(float)nk);

    if (n % 2 == 0) 
        median.push_back((distances[nk/2] + distances[nk/2 -1]) / 2.0);
    else median.push_back(distances[nk/2]);
     
}

int main(int argc, char ** argv) 
{

    int first_frame = -1, last_frame = -1, option = 0;
    char *in_frames_path = NULL, *out_frames_path = NULL,
         *stats_file = NULL, *cords_file = NULL;
    bool verbose = false, use_gpu = false;

    /* Get input arguments */ 
    while ((option = getopt(argc, argv,"s:e:p:c:f:o:vg")) != -1) 
    {
        switch (option) {
            case 's' : first_frame = atoi(optarg);
                break;
            case 'e' : last_frame = atoi(optarg);
                break;
            case 'p' : in_frames_path = optarg; 
                break;
            case 'c' : cords_file = optarg;
                break;
            case 'f' : stats_file = optarg;
                 break;
            case 'o' : out_frames_path = optarg;
                break;
            case 'v': verbose = true;
                break;
            case 'g': use_gpu = true;
                break;
            default:  
                exit(EXIT_FAILURE);
        }
    }

    /* Check mandatory parameters */
    if (first_frame == -1) {cout<<"Invalid first frame"<<endl; return 0;}
    if (last_frame == -1) {cout<<"Invalid last frame"<<endl; return 0;}
    if (!in_frames_path) {cout<<"Specify input frames path!"<<endl; return 0;}
    if (!cords_file) {cout<<"Specify coordinate log!"<<endl; return 0;}

    /* Input frames in CV_8U and float formats */
    Mat input, input_float;
    /* Current and previous frames for tracking. CV_8U and float */
    Mat prev, current, fprev, fcurrent;
    /* Tracking points: OpenCV and custom. Index 0: previous frame
     * Index 1: current frame
     */       
    vector<Point2f> opencv_points[2];
    vector<Point2f> custom_points[2];
    /* String version of paths */
    string out_frames_path_str = "", in_frames_path_str = "";
        
    in_frames_path_str.assign(in_frames_path);
    
    if (out_frames_path)
        out_frames_path_str.assign(out_frames_path);
    

    /* Redirect input from coordinates file */
    if (!freopen(cords_file,"r",stdin))
    {
        cout<<"Fatal error. Couldn't open coordinates file"<<endl; 
        return 1;
    }

    /* OpenCV tracker parameters */
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21);

    /* OpenPose x,y keypoints: 'ground_truth' coordinates */
    vector<int> x_truth, y_truth;
    vector<float> minim, maxim, mean, median;

    /* Estimated OpenPose points in the current frame with respect
     * to the original tracking frame
     */  
    vector<int> x_itp, y_itp;
    vector<float> x_of, y_of;
    /* Status vectors for Optical Flow */
    vector<char> custom_status;
    vector<uchar> opencv_status;
    /* Gaussian Pyramid */
    vector<cv::Mat> pyramidImagesPrevious;

    float fx, fy, conf;

    /* Iterate through all frames */
    for (auto i = 0; i < last_frame + 1; i++)
    {
        int persons;
        
        if (verbose && i >= first_frame)
            cout<<"Processing Frame "<<i<<endl;

        /* Get current frame file name */
        string number = ""; 
        char buffer[15];
        sprintf(buffer,"%d",i);
        string temp(buffer);
 
        if (i > 99) number = temp;
        else if (i > 9) number = "0" + temp;
        else number = "00" + temp;
        
        /* Load image */
        string img = in_frames_path_str + number + format_img; 
        string output_image = out_frames_path_str + number + format_img;        
        /* Get current frame in both fromats (CV_8U and float) */
        input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);                                      
        input.convertTo(input_float, CV_32F);

        cin>>persons;
        
	    x_truth.clear();
        y_truth.clear();

        /* Iterate through all keypoints and persons */
        for (int p = 0; p < persons; p++)
        {
    	    /* Add all keypoints */
    	    for (int k = 0; k < keypoints; k++)
    	    {	      	    
    	        cin>>fx>>fy>>conf;
                /* Keypoints with less than 1% confidence are discarded */
                if (conf < 0.01) continue;	
                x_truth.push_back((int)fx);
                y_truth.push_back((int)fy);
            }
	    }

        /* Set tracking points */
        if (i == first_frame)
        {
            fcurrent = input_float.clone();
            current = input;
            /* Set initial optical flow points to OpenPose first 
             * tracking frame
             */      
            points2crd(x_truth, y_truth, opencv_points[1]);
            points2crd(x_truth, y_truth, custom_points[1]);

            custom_points[0].resize(custom_points[1].size());

            x_itp.assign(x_truth.begin(), x_truth.end());
            y_itp.assign(y_truth.begin(), y_truth.end());
            x_of.assign(x_truth.begin(), x_truth.end());
            y_of.assign(y_truth.begin(), y_truth.end());

            custom_status.assign(custom_points[1].size(),0);
            opencv_status.assign(custom_points[1].size(),0); 
        } 
        /* Draw OF keypoint estimates for frames (s,e] */
        if (i > first_frame && i < last_frame + 1)
        {
            vector<float> err;

            /* Swap previous and current points and frames */
            std::swap(opencv_points[1], opencv_points[0]);   
            std::swap(custom_points[1], custom_points[0]);
            cv::swap(fcurrent, fprev);
            cv::swap(current, prev);

            fcurrent = input_float.clone();
            current = input;
            /* Estimate corresponding OpenPose points on the current
             * frame with respect to the original frame. Used only for 
             * statistics
             */       
            interpolate_next(x_itp, y_itp, x_truth, y_truth);

            calcOpticalFlowPyrLK(prev, current, opencv_points[0], 
                               opencv_points[1], 
                                opencv_status, err, winSize, 5, 
                                 termcrit, 0, 0.001);
            
            if (stats_file)
            {
                crd2points(x_of, y_of, opencv_points[1]);                         
                get_statistics(x_itp, y_itp, x_of, y_of, 
                               maxim, mean,minim,median);
                
                if (verbose)
                   cout<<"MAXIMUM: "<<maxim.back()<<",  MINIMUM: "<<minim.back()
                       <<", MEAN: "<<mean.back()<<", MEDIAN: "
                       <<median.back()<<endl; 
            }
            

        if (use_gpu)   
        { 
 	        pyramidalLKGpu(custom_points[0], custom_points[1],
                   	       custom_status, fprev, fcurrent, 5, 21);
        }    
        else
        {
            std::vector<cv::Mat> pyramidImagesCurrent;
            pyramidalLKCpu(custom_points[0], custom_points[1],
                           pyramidImagesPrevious, pyramidImagesCurrent,
                           custom_status, fprev, fcurrent, 5, 21); 

            pyramidImagesPrevious = pyramidImagesCurrent;
        }
            
        if (out_frames_path)
            draw_all(input_float,x_truth,y_truth, opencv_points[1], 
                     custom_points[1], output_image);
        }     
        else if (out_frames_path)
                drawKeyPoints(input_float, x_truth, y_truth,output_image);
    }
    /* Finish processing frames */

    /* If stats file argument was passed generate stats file */
    if (stats_file)
    {
        if (!freopen(stats_file,"w+",stdout))
        {
            cout<<"Fatal error, couldn't open stats file!"<<endl;
            return 1;
        }

        /* Dump keypoints */
        for (auto i = 0u; i < maxim.size(); i++)
        {
            cout<<minim[i]<<endl;
            cout<<median[i]<<endl;
            cout<<maxim[i]<<endl;
            cout<<mean[i]<<endl;
        }
    }    

    return 0;
}
