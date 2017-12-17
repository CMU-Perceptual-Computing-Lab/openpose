#ifndef OPENPOSE_TRACKING_LKPYRAMIDAL_HPP
#define OPENPOSE_TRACKING_LKPYRAMIDAL_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API void runLKPyramidal(std::vector<cv::Point2f> &coord_I,
                               std::vector<cv::Point2f> &coord_J,
                               cv::Mat &prev,
                               cv::Mat &next,
                               std::vector<char> &status,
                               int levels,
                               int patch_size = 5);
    // int lkpyramidal_gpu(cv::Mat &I, cv::Mat &J,int levels, int patch_size, 
    //                 std::vector<cv::Point2f> &ptsI, std::vector<cv::Point2f> &ptsJ,
    //                 std::vector<char> &status);
}

#endif // OPENPOSE_TRACKING_LKPYRAMIDAL_HPP
