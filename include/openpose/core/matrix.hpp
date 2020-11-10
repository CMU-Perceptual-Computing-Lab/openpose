#ifndef OPENPOSE_CORE_MAT_HPP
#define OPENPOSE_CORE_MAT_HPP

#include <memory> // std::shared_ptr
#include <openpose/core/macros.hpp>

namespace op
{
    // Convert from Matrix into cv::Mat. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // cv::Mat opMat = OP2CVMAT(cv::Mat());
    #define OP_OP2CVMAT(opMat) \
        (*((cv::Mat*)((opMat).getCvMat())))

    // Convert from Matrix into const cv::Mat. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // cv::Mat opMat = OP2CVCONSTMAT(cv::Mat());
    #define OP_OP2CVCONSTMAT(opMat) \
        (*((cv::Mat*)((opMat).getConstCvMat())))

    // Convert from cv::Mat into Matrix. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // Matrix opMat = CV2OPMAT(Matrix());
    #define OP_CV2OPMAT(cvMat) \
        (op::Matrix((void*)&(cvMat)))

    // Convert from cv::Mat into const Matrix. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // Matrix opMat = CV2OPCONSTMAT(Matrix());
    #define OP_CV2OPCONSTMAT(cvMat) \
        (op::Matrix((const void*)&(cvMat)))

    // Convert from std::vector<Matrix> into std::vector<cv::Mat>. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // std::vector<Matrix> opMats; // Assume filled
        // OP_OP2CVVECTORMAT(cvMats, opMats);
    #define OP_OP2CVVECTORMAT(cvMats, opMats) \
        std::vector<cv::Mat> cvMats; \
        for (auto& opMat : (opMats)) \
        { \
            const auto cvMat = OP_OP2CVCONSTMAT(opMat); \
            cvMats.emplace_back(cvMat); \
        }

    // Convert from std::vector<cv::Mat> into std::vector<Matrix>. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // std::vector<cv::Mat> cvMats; // Assume filled
        // OP_CV2OPVECTORMAT(opMats, cvMats);
    #define OP_CV2OPVECTORMAT(opMats, cvMats) \
        std::vector<op::Matrix> opMats; \
        for (auto& cvMat : (cvMats)) \
        { \
            const auto opMat = OP_CV2OPMAT(cvMat); \
            opMats.emplace_back(opMat); \
        }

    // Convert from std::vector<cv::Mat> into std::vector<Matrix>. Usage example:
        // #include <opencv2/core/core.hpp>
        // ...
        // // Equivalents:
        // OP_CV_VOID_FUNCTION(opMat, size());
        // // and
        // OP_OP2CVMAT(cvMat, opMat);
        // cvMat.size();
    #define OP_MAT_VOID_FUNCTION(opMat, function) \
        { \
            cv::Mat cvMat = OP_OP2CVMAT(cvMat, opMat); \
            cvMat.function; \
        }
    #define OP_CONST_MAT_VOID_FUNCTION(opMat, function) \
        { \
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(opMat); \
            cvMat.function; \
        }
    #define OP_MAT_RETURN_FUNCTION(outputVariable, opMat, function) \
        { \
            cv::Mat cvMat = OP_OP2CVMAT(cvMat, opMat); \
            outputVariable = cvMat.function; \
        }
    #define OP_CONST_MAT_RETURN_FUNCTION(outputVariable, opMat, function) \
        { \
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(opMat); \
            outputVariable = cvMat.function; \
        }

    /**
     * Matrix: Bind of cv::Mat to avoid OpenCV as dependency in the headers.
     */
    class OP_API Matrix
    {
    public:
        /**
         * @param matrixesResized For 3-D OpenPose, if >1, it will assume the image is composed of
         * numberImagesStackedHorizontally horizontally stacked images. It must be already resized to avoid
         * internally allocating/removing elements of std::vector (to avoid errors if using different std DLLs)
         * @param cvMatPtr should be a cv::Mat element or it will provoke a core dumped. Done to
         * avoid explicitly exposing 3rdparty libraries on the headers.
         */
        static void splitCvMatIntoVectorMatrix(std::vector<Matrix>& matrixesResized, const void* const cvMatPtr);

        Matrix();

        /**
         * @param cvMatPtr should be a cv::Mat element or it will provoke a core dumped. Done to
         * avoid explicitly exposing 3rdparty libraries on the headers.
         */
        explicit Matrix(const void* cvMatPtr);

        /**
         * Analog to cv::Mat(int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
         */
        explicit Matrix(const int rows, const int cols, const int type);

        /**
         * Analog to cv::Mat(int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
         * Very important: This Matrix will only "borrow" this pointer, so the caller must make sure to maintain the
         * memory allocated until this Matrix destructor is called and also to handle the ucharPtr memory deallocation.
         * @param ucharPtr should be a cv::Mat::data (or analog) element or it will provoke a core dumped. Done to
         * avoid explicitly exposing 3rdparty libraries on the headers.
         */
        explicit Matrix(const int rows, const int cols, const int type, void* cvMatPtr);

        Matrix clone() const;

        /**
         * @return cv::Mat*.
         */
        void* getCvMat();

        /**
         * @return const cv::Mat*.
         */
        const void* getConstCvMat() const;

        /**
         * Equivalent to cv::Mat::data
         * @return A raw pointer to the internal data of cv::Mat.
         */
        unsigned char* data();
        /**
         * Equivalent to cv::Mat::data
         * @return A raw pointer to the internal data of cv::Mat.
         */
        const unsigned char* dataConst() const;
        /**
         * Similar to dataConst(), but it allows the data to be edited.
         * This function is only implemented for Pybind11 usage.
         * @return A raw pointer to the internal data of cv::Mat.
         */
        unsigned char* dataPseudoConst() const;

        /**
         * Equivalent to cv::Mat::eye
         */
        static Matrix eye(const int rows, const int cols, const int type);
        /**
         * Equivalent to cv::Mat::cols
         */
        int cols() const;
        /**
         * Equivalent to cv::Mat::rows
         */
        int rows() const;
        /**
         * Equivalent to cv::Mat::size[dimension]
         */
        int size(const int dimension) const;
        /**
         * Equivalent to cv::Mat::dims
         */
        int dims() const;

        /**
         * Equivalent to their analog cv::Mat functions
         */
        bool isContinuous() const;
        bool isSubmatrix() const;
        size_t elemSize() const;
        size_t elemSize1() const;
        int type() const;
        int depth() const;
        int channels() const;
        size_t step1(const int i = 0) const;
        bool empty() const;
        size_t total() const;
        int checkVector(const int elemChannels, const int depth = -1, const bool requireContinuous = true) const;

        /**
         * Similar to their analog cv::Mat functions
         */
        void setTo(const double value);
        void copyTo(Matrix& outputMat) const;

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplMatrix;
        std::shared_ptr<ImplMatrix> spImpl;
    };
}

#endif // OPENPOSE_CORE_MAT_HPP
