// #include <iostream>
#include <opencv2/core/core.hpp> // cv::Point2f, cv::Mat
#include <opencv2/imgproc/imgproc.hpp> // cv::pyrDown
#include <opencv2/video/video.hpp> // cv::buildOpticalFlowPyramid
#include <openpose/utilities/profiler.hpp>
#include <openpose/tracking/pyramidalLK.hpp>

#if defined (WITH_SSE4)
#include <emmintrin.h>
#include "smmintrin.h"
#endif

#if defined (WITH_AVX)
#include <immintrin.h>
#endif

#include <iostream>
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
#if defined (WITH_SSE4)
    float sse_dot_product(std::vector<float> &av, std::vector<float> &bv)
    {

      /* Get SIMD-vector pointers to the start of each vector */
      unsigned int niters = av.size() / 4;
      float zeros[] = {0.0, 0.0, 0.0, 0.0};

      float *a = (float *) aligned_alloc(16, av.size()*sizeof(float));
      float *b = (float *) aligned_alloc(16, av.size()*sizeof(float));
      memcpy(a,&av[0],av.size()*sizeof(float));
      memcpy(b,&bv[0],bv.size()*sizeof(float));

      __m128 *ptrA = (__m128*) &a[0], *ptrB = (__m128*) &b[0];
      __m128 res = _mm_load_ps(zeros);

      /* Do SIMD dot product */
      for (unsigned int i = 0; i < niters; i++, ptrA++,ptrB++)
        res = _mm_add_ps(_mm_dp_ps(*ptrA, *ptrB, 255), res);
      

      /* Get result back from the SIMD vector */
      float fres[4];
      _mm_store_ps (fres, res);
      int q = 4 * niters;

      for (unsigned int i = 0; i < av.size() % 4; i++)
        fres[0] += (a[i+q]*b[i+q]);

      free(a);
      free(b);

      return fres[0];
    }
#endif

#if defined (WITH_AVX)

    float avx_dot_product(std::vector<float> &av, std::vector<float> &bv)
    {

      /* Get SIMD-vector pointers to the start of each vector */
      unsigned int niters = av.size() / 8;

      float *a = (float *) aligned_alloc(32, av.size()*sizeof(float));
      float *b = (float *) aligned_alloc(32, av.size()*sizeof(float));
      memcpy(a,&av[0],av.size()*sizeof(float));
      memcpy(b,&bv[0],bv.size()*sizeof(float));

      __m256 *ptrA = (__m256*) &a[0], *ptrB = (__m256*) &b[0];
      __m256 res = _mm256_set1_ps(0.0);

      for (unsigned int i = 0; i < niters; i++, ptrA++,ptrB++)
        res = _mm256_add_ps(_mm256_dp_ps(*ptrA, *ptrB, 255), res);

      /* Get result back from the SIMD vector */
      float fres[8];
      _mm256_storeu_ps (fres, res);
      int q = 8 * niters;

      for (unsigned int i = 0; i < av.size() % 8; i++)
        fres[0] += (a[i+q]*b[i+q]);

      free(a);
      free(b);

      return fres[0] + fres[4];
    }
#endif 

    char computeLK(cv::Point2f& delta,  std::vector<float>& ix,
                  std::vector<float>& iy, std::vector<float>& it)
    {
        try
        {
            // Calculate sums
            auto sumXX = 0.f;
            auto sumYY = 0.f;
            auto sumXT = 0.f;
            auto sumYT = 0.f;
            auto sumXY = 0.f;

#if defined (WITH_AVX)
            sumXX = avx_dot_product(ix,ix);
            sumYY = avx_dot_product(iy,iy);
            sumXY = avx_dot_product(ix,iy);
            sumXT = avx_dot_product(ix,it);
            sumYT = avx_dot_product(iy,it);
#elif defined (WITH_SSE4)
            sumXX = sse_dot_product(ix,ix);
            sumYY = sse_dot_product(iy,iy);
            sumXY = sse_dot_product(ix,iy);
            sumXT = sse_dot_product(ix,it);
            sumYT = sse_dot_product(iy,it);
#else            
            for (auto i = 0u; i < ix.size(); i++)
            {
              sumXX += ix[i] * ix[i];
              sumYY += iy[i] * iy[i];
              sumXY += ix[i] * iy[i];
              sumXT += ix[i] * it[i];
              sumYT += iy[i] * it[i];
            }            
#endif            

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
                    ix[baseIndex+j-1] = (patch[i][j+1] - patch[i][j-1])/2.f;
                    iy[baseIndex+j-1] = (patch[i+1][j] - patch[i-1][j])/2.f;
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

    char extractPatchIt(std::vector<std::vector<float>>& patch, const int xI, const int yI, const int xJ,
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
            // if (status)
            //     return result;

            status = extractPatchIt(patchIt, int(pointI.x), int(pointI.y), int(pointJ.x), int(pointJ.y), I, J, patchSize);

            // if (status)
            //     return result;

            // Get the Ix, Iy and It vectors
            std::vector<float> ix, iy, it;
            getVectors(ix, iy, it, patch, patchIt, patchSize);

            // Calculate optical flow
            cv::Point2f delta;
            status = computeLK(delta, ix, iy, it);

            // if (status)
            //     return result;

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

    // old, nwekp, pyramidPrev, pyramidCurr, status, imagePrev, imageCurr
    void pyramidalLKOcv(std::vector<cv::Point2f>& coordI, std::vector<cv::Point2f>& coordJ,
                        std::vector<cv::Mat>& pyramidImagesPrevious, std::vector<cv::Mat>& pyramidImagesCurrent,
                        std::vector<char>& status, const cv::Mat& imagePrevious,
                        const cv::Mat& imageCurrent, const int levels, const int patchSize, const bool initFlow)
    {
        try
        {
            // Empty coordinates
            if (coordI.size() != 0)
            {
                // const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);

                std::vector<cv::Point2f> I;
                I.assign(coordI.begin(), coordI.end());

                if (!initFlow)
                {
                    coordJ.clear();
                    coordJ.assign(I.begin(), I.end());
                }

                const cv::Mat& imagePrevGray = imagePrevious;
                const cv::Mat& imageCurrGray = imageCurrent;

                // Compute Pyramids
                if (pyramidImagesPrevious.empty())
                    cv::buildOpticalFlowPyramid(imagePrevGray, pyramidImagesPrevious, cv::Size{patchSize,patchSize}, levels);
                if (pyramidImagesCurrent.empty())
                    cv::buildOpticalFlowPyramid(imageCurrGray, pyramidImagesCurrent, cv::Size{patchSize,patchSize}, levels);

                // Compute Flow
                std::vector<uchar> st;
                std::vector<float> err;
                if (initFlow)
                    cv::calcOpticalFlowPyrLK(pyramidImagesPrevious, pyramidImagesCurrent, coordI, coordJ, st, err,
                                             cv::Size{patchSize,patchSize},levels,
                                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,30,0.01),
                                             cv::OPTFLOW_USE_INITIAL_FLOW);
                else
                    cv::calcOpticalFlowPyrLK(pyramidImagesPrevious, pyramidImagesCurrent, coordI, coordJ, st, err,
                                             cv::Size{patchSize,patchSize},levels);

                // Check distance
                for (size_t i=0; i<status.size(); i++)
                {
                    const float distance = std::sqrt(
                        std::pow(coordI[i].x-coordJ[i].x,2) + std::pow(coordI[i].y-coordJ[i].y,2));

                    // Check if lk loss track, if distance is close keep it
                    if (st[i] != (status[i]))
                        if (distance <= patchSize*2)
                            st[i] = 1;

                    // If distance too far discard it
                    if (distance > patchSize*2)
                        st[i] = 0;
                }

                // Stupid hack because apparently in this tracker 0 means 1 and 1 is 0 wtf
                if (st.size() != status.size())
                    error("st.size() != status.size().", __LINE__, __FUNCTION__, __FILE__);
                for (size_t i=0; i<status.size(); i++)
                {
                    // If its 0 to begin with (Because OP lost track?)
                    if (status[i] != 0)
                    {
                        if (st[i] == 0)
                            st[i] = 0;
                        else if (st[i] == 1)
                            st[i] = 1;
                        else
                            error("Wrong CV Type.", __LINE__, __FUNCTION__, __FILE__);
                        status[i] = st[i];
                    }
                }

                // Profiler::timerEnd(profilerKey);
                // Profiler::printAveragedTimeMsEveryXIterations(profilerKey, __LINE__, __FUNCTION__, __FILE__, 5);

                // // Debug
                // std::cout << "LK: ";
                // for (int i=0; i<status.size(); i++) std::cout << !(int)status[i];
                // std::cout << std::endl;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
