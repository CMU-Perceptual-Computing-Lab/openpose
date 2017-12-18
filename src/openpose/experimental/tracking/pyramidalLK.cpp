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
#include <openpose/experimental/tracking/pyramidalLK.hpp>
//#define DEBUG

// #ifdef DEBUG
// // When debugging is enabled, these form aliases to useful functions
// #define dbg_printf(...) printf(__VA_ARGS__);
// #else
// // When debugging is disabled, no code gets generated for these
// #define dbg_printf(...)
// #endif

#define SUCCESS 0
#define INVALID_PATCH_SIZE 1
#define OUT_OF_FRAME 2
#define ZERO_DENOMINATOR 3

// using namespace cv;
// using namespace std;



namespace op
{
    char computeLK(cv::Point2f& delta, const std::vector<float>& ix,
                   const std::vector<float>& iy, const std::vector<float>& it)
    {
        auto sumXX = 0.f;
        auto sumYY = 0.f;
        auto sumXT = 0.f;
        auto sumYT = 0.f;
        auto sumXY = 0.f;
        float numU, numV, den, u, v;

        // Calculate sums
        for (auto i = 0u; i < ix.size(); i++)
        {
            sumXX += ix[i] * ix[i];
            sumYY += iy[i] * iy[i];
            sumXY += ix[i] * iy[i];
            sumXT += ix[i] * it[i];
            sumYT += iy[i] * it[i];
        }

        // Get numerator and denominator of u and v
        den = (sumXX*sumYY) - (sumXY * sumXY);

        if (std::abs(den) < 1e-9f)
            return ZERO_DENOMINATOR;

        numU = (-1.0 * sumYY * sumXT) + (sumXY * sumYT);
        numV = (-1.0 * sumXX * sumYT) + (sumXT * sumXY);

        u = numU / den;
        v = numV / den;
        delta.x = u;
        delta.y = v;

        return SUCCESS;
    }

    void getVectors(std::vector<float>& ix, std::vector<float>& iy, std::vector<float>& it,
                    const std::vector<std::vector<float>>& patch, const std::vector<std::vector<float>>& patchIt,
                    const int patchSize)
    {
        for (auto i = 1; i <= patchSize; i++)
        {
            for (auto j = 1; j <= patchSize; j++)
            {
                ix.push_back((patch[i][j+1] - patch[i][j-1])/2.0);
                iy.push_back((patch[i+1][j] - patch[i-1][j])/2.0);
            }
        }

        for (auto i = 0; i < patchSize; i++)
            for (auto j = 0; j < patchSize; j++)
                it.push_back(patchIt[i][j]);

    }

    char extractPatch(std::vector< std::vector<float>>& patch, const int x, const int y, const int patchSize,
                      const cv::Mat& image)
    {
        int radix = patchSize / 2;

        if ( ((x - radix) < 0) ||
             ((x + radix) >= image.cols) ||
             ((y - radix) < 0) ||
             ((y + radix) >= image.rows))
            return OUT_OF_FRAME;

        for (auto i = -radix; i <= radix; i++)
            for (auto j = -radix; j <= radix; j++)
                patch[i+radix][j+radix] = image.at<float>(y+i,x+j);

        return SUCCESS;
    }

    char extractPatchIt(std::vector< std::vector<float>>& patch, const int xI, const int yI, const int xJ,
                        const int yJ, const cv::Mat& I, const cv::Mat& J, const int patchSize)
    {
        const int radix = patchSize / 2;

        if (((xI - radix) < 0) ||
             ((xI + radix) >= I.cols) ||
             ((yI - radix) < 0) ||
             ((yI + radix) >= I.rows))
            return OUT_OF_FRAME;

        if (((xJ - radix) < 0) ||
             ((xJ + radix) >= J.cols) ||
             ((yJ - radix) < 0) ||
             ((yJ + radix) >= J.rows))
            return OUT_OF_FRAME;

        for (auto i = -radix; i <= radix; i++)
            for (auto j = -radix; j <= radix; j++)
                patch[i+radix][j+radix] = J.at<float>(yJ+i,xJ+j) - I.at<float>(yI+i,xI+j);

        return SUCCESS;

    }
    // Given an OpenCV image, build a gaussian pyramid of size 'levels'
    void buildGaussianPyramid(std::vector<cv::Mat>& pyramidImages, const cv::Mat& image, const int levels)
    {
        pyramidImages.clear();
        pyramidImages.emplace_back(image);

        for (auto i = 0; i < levels - 1; i++)
        {
            cv::Mat pyredImage;
            cv::pyrDown(pyramidImages.back(), pyredImage);
            pyramidImages.emplace_back(pyredImage);
        }
    }



    cv::Point2f pyramidIteration(char& status, const cv::Point2f& pointI, const cv::Point2f& pointJ, const cv::Mat& I,
                                 const cv::Mat& J, const int patchSize = 5)
    {

        cv::Point2f result;

        // Extract a patch around the image
        std::vector<std::vector<float>> patch(patchSize + 2, std::vector<float>(patchSize + 2));
        std::vector<std::vector<float>> patchIt(patchSize, std::vector<float>(patchSize));

        status = extractPatch(patch, (int)pointI.x,(int)pointI.y, patchSize + 2, I);
    //    if (status)
    //        return result;

        status = extractPatchIt(patchIt, pointI.x, pointI.y, pointJ.x, pointJ.y, I, J, patchSize);

    //    if (status)
    //        return result;

        // Get the Ix, Iy and It vectors
        std::vector<float> ix, iy, it;
        getVectors(ix, iy, it, patch, patchIt, patchSize);

        // Calculate optical flow
        cv::Point2f delta;
        status = computeLK(delta, ix, iy, it);

    //    if (status)
    //        return result;

        result = pointJ + delta;

        return result;
    }

    void pyramidalLKCpu(std::vector<cv::Point2f>& coordI, std::vector<cv::Point2f>& coordJ,
                        std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
                        std::vector<char>& status, const cv::Mat& imagePrevious,
                        const cv::Mat& imageCurrent, const int levels, const int patchSize)
    {
        // Empty coordinates
        if (coordI.size() == 0)
            return;

        std::vector<cv::Point2f> I;
        I.assign(coordI.begin(), coordI.end());

        const auto rescaleScale = 1.0/(float)(1<<(levels-1));
        for (auto& coordenate : I)
            coordenate *= rescaleScale;

        coordJ.clear();
        coordJ.assign(I.begin(), I.end());

        if (pyramidImagesPrevious.empty())
            buildGaussianPyramid(pyramidImagesPrevious, imagePrevious, levels);
        if (pyramidImagesCurrent.empty())
            buildGaussianPyramid(pyramidImagesCurrent, imageCurrent, levels);

        // Process all pixel requests
        for (auto i = 0u; i < coordI.size(); i++)
        {
            for (auto l = levels - 1; l >= 0; l--)
            {
                char status_point = 0;
                cv::Point2f result;

                result = pyramidIteration(status_point, I[i], coordJ[i],pyramidImagesPrevious[l],
                                          pyramidImagesCurrent[l], patchSize);
                if (status_point)
                    status[i] = status_point;

                coordJ[i] = result;

                if (l == 0)
                    break;

                I[i] *= 2.f;
                coordJ[i] *= 2.f;
            }
        }
    }
}
