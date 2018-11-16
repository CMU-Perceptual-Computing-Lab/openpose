#ifndef OPENPOSE_CORE_MACROS_HPP
#define OPENPOSE_CORE_MACROS_HPP

#include <chrono> // std::chrono:: functionaligy, e.g., std::chrono::milliseconds
#include <memory> // std::shared_ptr
#include <ostream>
#include <string>
#include <thread> // std::this_thread
#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat, check OpenCV version

// OpenPose name and version
const std::string OPEN_POSE_NAME_STRING = "OpenPose";
const std::string OPEN_POSE_VERSION_STRING = "1.4.0";
const std::string OPEN_POSE_NAME_AND_VERSION = OPEN_POSE_NAME_STRING + " " + OPEN_POSE_VERSION_STRING;
// #define COMMERCIAL_LICENSE

#ifndef _WIN32
    #define OP_API
#elif defined OP_EXPORTS
    #define OP_API __declspec(dllexport)
#else
    #define OP_API __declspec(dllimport)
#endif

// Disable some Windows Warnings
#ifdef _WIN32
    #pragma warning ( disable : 4251 ) // XXX needs to have dll-interface to be used by clients of class YYY
    #pragma warning( disable: 4275 ) // non dll-interface structXXX used as base
#endif

#define UNUSED(unusedVariable) (void)(unusedVariable)

#define DELETE_COPY(className) \
    className(const className&) = delete; \
    className& operator=(const className&) = delete

// Instantiate a class with all the basic types
#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) COMPILE_TEMPLATE_BASIC_TYPES(className, class)
#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) COMPILE_TEMPLATE_BASIC_TYPES(className, struct)
#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
    template classType OP_API className<char>; \
    template classType OP_API className<signed char>; \
    template classType OP_API className<short>; \
    template classType OP_API className<int>; \
    template classType OP_API className<long>; \
    template classType OP_API className<long long>; \
    template classType OP_API className<unsigned char>; \
    template classType OP_API className<unsigned short>; \
    template classType OP_API className<unsigned int>; \
    template classType OP_API className<unsigned long>; \
    template classType OP_API className<unsigned long long>; \
    template classType OP_API className<float>; \
    template classType OP_API className<double>; \
    template classType OP_API className<long double>

/**
 * cout operator overload calling toString() function
 * @return std::ostream containing output from toString()
 */
#define OVERLOAD_C_OUT(className) \
    template<typename T> std::ostream &operator<<(std::ostream& ostream, const op::className<T>& obj) \
    { \
        ostream << obj.toString(); \
        return ostream; \
    }

// Instantiate a class with float and double specifications
#define COMPILE_TEMPLATE_FLOATING_TYPES_CLASS(className) COMPILE_TEMPLATE_FLOATING_TYPES(className, class)
#define COMPILE_TEMPLATE_FLOATING_TYPES_STRUCT(className) COMPILE_TEMPLATE_FLOATING_TYPES(className, struct)
#define COMPILE_TEMPLATE_FLOATING_TYPES(className, classType) \
  char gInstantiationGuard##className; \
  template classType OP_API className<float>; \
  template classType OP_API className<double>

// PIMPL does not work if function arguments need the 3rd-party class. Alternative:
// stackoverflow.com/questions/13978775/how-to-avoid-include-dependency-to-external-library?answertab=active#tab-top
struct dim3;
namespace caffe
{
    template <typename T> class Blob;
}
namespace boost
{
    template <typename T> class shared_ptr; // E.g., boost::shared_ptr<caffe::Blob<float>>
}

// Compabitility for OpenCV 4.0 while preserving 2.4.X and 3.X compatibility
// Note:
// - CV_VERSION:         2.4.9.1 | 4.0.0-beta
// - CV_MAJOR_VERSION:         2 | 4
// - CV_MINOR_VERSION:         4 | 0
// - CV_SUBMINOR_VERSION:      9 | 0
// - CV_VERSION_EPOCH:         2 | Not defined
#if (defined(CV_MAJOR_VERSION) && CV_MAJOR_VERSION > 3)
    #define OPEN_CV_IS_4_OR_HIGHER
#endif
#ifdef OPEN_CV_IS_4_OR_HIGHER
    #define CV_BGR2GRAY cv::COLOR_BGR2GRAY
    #define CV_CALIB_CB_ADAPTIVE_THRESH cv::CALIB_CB_ADAPTIVE_THRESH
    #define CV_CALIB_CB_NORMALIZE_IMAGE cv::CALIB_CB_NORMALIZE_IMAGE
    #define CV_CALIB_CB_FILTER_QUADS cv::CALIB_CB_FILTER_QUADS
    #define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
    #define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
    #define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
    #define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
    #define CV_CAP_PROP_POS_FRAMES cv::CAP_PROP_POS_FRAMES
    #define CV_FOURCC cv::VideoWriter::fourcc
    #define CV_GRAY2BGR cv::COLOR_GRAY2BGR
    #define CV_HAAR_SCALE_IMAGE cv::CASCADE_SCALE_IMAGE
    #define CV_INTER_CUBIC cv::INTER_CUBIC
    #define CV_INTER_LINEAR cv::INTER_LINEAR
    #define CV_L2 cv::NORM_L2
    #define CV_TERMCRIT_EPS cv::TermCriteria::Type::EPS
    #define CV_TERMCRIT_ITER cv::TermCriteria::Type::MAX_ITER
    #define CV_WARP_INVERSE_MAP cv::WARP_INVERSE_MAP
    #define CV_WINDOW_FULLSCREEN cv::WINDOW_FULLSCREEN
    #define CV_WINDOW_KEEPRATIO cv::WINDOW_KEEPRATIO
    #define CV_WINDOW_NORMAL cv::WINDOW_NORMAL
    #define CV_WINDOW_OPENGL cv::WINDOW_OPENGL
    #define CV_WND_PROP_FULLSCREEN cv::WND_PROP_FULLSCREEN
    // Required for alpha and beta versions, but not for rc version
    #include <opencv2/imgcodecs/imgcodecs.hpp>
    #ifndef CV_IMWRITE_JPEG_QUALITY
        #define CV_IMWRITE_JPEG_QUALITY cv::IMWRITE_JPEG_QUALITY
    #endif
    #ifndef CV_IMWRITE_PNG_COMPRESSION
        #define CV_IMWRITE_PNG_COMPRESSION cv::IMWRITE_PNG_COMPRESSION
    #endif
    #ifndef CV_LOAD_IMAGE_ANYDEPTH
        #define CV_LOAD_IMAGE_ANYDEPTH cv::IMREAD_ANYDEPTH
    #endif
    #ifndef CV_LOAD_IMAGE_COLOR
        #define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
    #endif
    #ifndef CV_LOAD_IMAGE_GRAYSCALE
        #define CV_LOAD_IMAGE_GRAYSCALE cv::IMREAD_GRAYSCALE
    #endif
#endif

#endif // OPENPOSE_CORE_MACROS_HPP
