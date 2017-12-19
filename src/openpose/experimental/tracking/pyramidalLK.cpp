// #include <iostream>
#include <opencv2/core/core.hpp> // cv::Point2f, cv::Mat
#include <opencv2/imgproc/imgproc.hpp> // cv::pyrDown
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
#define UNDEFINED_ERROR 4

namespace op
{
    char computeLK(cv::Point2f& delta, const std::vector<float>& ix,
                   const std::vector<float>& iy, const std::vector<float>& it)
    {
        try
        {
            // Calculate sums
            auto sumXX = 0.f;
            auto sumYY = 0.f;
            auto sumXT = 0.f;
            auto sumYT = 0.f;
            auto sumXY = 0.f;
            for (auto i = 0u; i < ix.size(); i++)
            {
                sumXX += ix[i] * ix[i];
                sumYY += iy[i] * iy[i];
                sumXY += ix[i] * iy[i];
                sumXT += ix[i] * it[i];
                sumYT += iy[i] * it[i];
            }

            // Get numerator and denominator of u and v
            const auto den = (sumXX*sumYY) - (sumXY * sumXY);

            if (std::abs(den) < 1e-9f)
                return ZERO_DENOMINATOR;

            const auto numU = (-1.f * sumYY * sumXT) + (sumXY * sumYT);
            const auto numV = (-1.f * sumXX * sumYT) + (sumXT * sumXY);

            delta.x = numU / den;
            delta.y = numV / den;

            return SUCCESS;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return UNDEFINED_ERROR;
        }
    }

    void getVectors(std::vector<float>& ix, std::vector<float>& iy, std::vector<float>& it,
                    const std::vector<std::vector<float>>& patch, const std::vector<std::vector<float>>& patchIt,
                    const int patchSize)
    {
        try
        {
            // Performance: resize faster than emplace_back/push_back
            const auto numberElements = patchSize*patchSize;
            // Get `ix` and `iy`
            ix.resize(numberElements);
            iy.resize(numberElements);
            for (auto i = 1; i <= patchSize; i++)
            {
                const auto baseIndex = (i-1)*patchSize;
                for (auto j = 1; j <= patchSize; j++)
                {
                    ix[baseIndex+j-1] = (patch[i][j+1] - patch[i][j-1])/2.0;
                    iy[baseIndex+j-1] = (patch[i+1][j] - patch[i-1][j])/2.0;
                }
            }
            // Get `it`
            it.resize(numberElements);
            for (auto i = 0; i < patchSize; i++)
            {
                const auto baseIndex = i*patchSize;
                for (auto j = 0; j < patchSize; j++)
                    it[baseIndex+j] = patchIt[i][j];
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    char extractPatch(std::vector< std::vector<float>>& patch, const int x, const int y, const int patchSize,
                      const cv::Mat& image)
    {
        try
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
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return UNDEFINED_ERROR;
        }
    }

    char extractPatchIt(std::vector< std::vector<float>>& patch, const int xI, const int yI, const int xJ,
                        const int yJ, const cv::Mat& I, const cv::Mat& J, const int patchSize)
    {
        try
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
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return UNDEFINED_ERROR;
        }
    }
    // Given an OpenCV image, build a gaussian pyramid of size 'levels'
    void buildGaussianPyramid(std::vector<cv::Mat>& pyramidImages, const cv::Mat& image, const int levels)
    {
        try
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
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }



    cv::Point2f pyramidIteration(char& status, const cv::Point2f& pointI, const cv::Point2f& pointJ, const cv::Mat& I,
                                 const cv::Mat& J, const int patchSize = 5)
    {
        try
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
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return cv::Point2f{};
        }
    }

    void pyramidalLKCpu(std::vector<cv::Point2f>& coordI, std::vector<cv::Point2f>& coordJ,
                        std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
                        std::vector<char>& status, const cv::Mat& imagePrevious,
                        const cv::Mat& imageCurrent, const int levels, const int patchSize)
    {
        try
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
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
