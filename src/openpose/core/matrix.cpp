#include <openpose/core/matrix.hpp>
#include <opencv2/core/core.hpp> // cv::Mat
#include <openpose/utilities/errorAndLog.hpp>

namespace op
{
    struct Matrix::ImplMatrix
    {
        cv::Mat mCvMat;
    };

    void Matrix::splitCvMatIntoVectorMatrix(std::vector<Matrix>& matrixesResized, const void* const cvMatPtr)
    {
        try
        {
            const auto numberImagesStackedHorizontally = matrixesResized.size();
            // Sanity check
            if (numberImagesStackedHorizontally < 1)
                error("matrixesResized.size() must be greater than 0.", __LINE__, __FUNCTION__, __FILE__);
            // Split cv::Mat
            cv::Mat matConcatenated = *((cv::Mat*) cvMatPtr);
            const auto individualWidth = matConcatenated.cols/numberImagesStackedHorizontally;
            for (auto i = 0u ; i < numberImagesStackedHorizontally ; i++)
            {
                cv::Mat cvMat(
                    matConcatenated,
                    cv::Rect{
                        (int)(i*individualWidth), 0,
                        (int)individualWidth, (int)matConcatenated.rows });
                matrixesResized[i] = OP_CV2OPMAT(cvMat);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix::Matrix() :
        spImpl{std::make_shared<ImplMatrix>()}
    {
    }

    Matrix::Matrix(const void* cvMatPtr) :
        spImpl{std::make_shared<ImplMatrix>()}
    {
        try
        {
            spImpl->mCvMat = *((cv::Mat*) cvMatPtr);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix::Matrix(const int rows, const int cols, const int type) :
        spImpl{std::make_shared<ImplMatrix>()}
    {
        try
        {
            spImpl->mCvMat = cv::Mat(rows, cols, type);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix::Matrix(const int rows, const int cols, const int type, void* cvMatPtr) :
        spImpl{std::make_shared<ImplMatrix>()}
    {
        try
        {
            spImpl->mCvMat = cv::Mat(rows, cols, type, cvMatPtr);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix Matrix::clone() const
    {
        try
        {
            Matrix matrix;
            matrix.spImpl->mCvMat = spImpl->mCvMat.clone();
            return matrix;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    void* Matrix::getCvMat()
    {
        try
        {
            return (void*)(&spImpl->mCvMat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const void* Matrix::getConstCvMat() const
    {
        try
        {
            return (const void*)(&spImpl->mCvMat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    unsigned char* Matrix::data()
    {
        try
        {
            return spImpl->mCvMat.data;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const unsigned char* Matrix::dataConst() const
    {
        try
        {
            return spImpl->mCvMat.data;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    unsigned char* Matrix::dataPseudoConst() const
    {
        try
        {
            return spImpl->mCvMat.data;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    Matrix Matrix::eye(const int rows, const int cols, const int type)
    {
        try
        {
            Matrix matrix;
            matrix.spImpl->mCvMat = cv::Mat::eye(rows, cols, type);
            return matrix;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }

    int Matrix::cols() const
    {
        try
        {
            return spImpl->mCvMat.cols;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int Matrix::rows() const
    {
        try
        {
            return spImpl->mCvMat.rows;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int Matrix::size(const int dimension) const
    {
        try
        {
            return spImpl->mCvMat.size[dimension];
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int Matrix::dims() const
    {
        try
        {
            return spImpl->mCvMat.dims;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    bool Matrix::isContinuous() const
    {
        try
        {
            return spImpl->mCvMat.isContinuous();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    bool Matrix::isSubmatrix() const
    {
        try
        {
            return spImpl->mCvMat.isSubmatrix();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }
    size_t Matrix::elemSize() const
    {
        try
        {
            return spImpl->mCvMat.elemSize();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return size_t(-1);
        }
    }

    size_t Matrix::elemSize1() const
    {
        try
        {
            return spImpl->mCvMat.elemSize();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return size_t(-1);
        }
    }

    int Matrix::type() const
    {
        try
        {
            return spImpl->mCvMat.type();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    int Matrix::depth() const
    {
        try
        {
            return spImpl->mCvMat.depth();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }


    int Matrix::channels() const
    {
        try
        {
            return spImpl->mCvMat.channels();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    size_t Matrix::step1(const int i) const
    {
        try
        {
            return spImpl->mCvMat.step1(i);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return size_t(-1);
        }
    }

    bool Matrix::empty() const
    {
        try
        {
            return spImpl->mCvMat.empty();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    size_t Matrix::total() const
    {
        try
        {
            return spImpl->mCvMat.total();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return size_t(-1);
        }
    }

    int Matrix::checkVector(const int elemChannels, const int depth, const bool requireContinuous) const
    {
        try
        {
            return spImpl->mCvMat.checkVector(elemChannels, depth, requireContinuous);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    void Matrix::setTo(const double value)
    {
        try
        {
            spImpl->mCvMat.setTo(value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Matrix::copyTo(Matrix& outputMat) const
    {
        try
        {
            spImpl->mCvMat.copyTo(outputMat.spImpl->mCvMat);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
