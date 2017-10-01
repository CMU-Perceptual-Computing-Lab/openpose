#include <openpose/utilities/openCv.hpp>
#include <openpose/core/cvMatToOpOutput.hpp>

namespace op
{
    CvMatToOpOutput::CvMatToOpOutput(const Point<int>& outputResolution, const bool generateOutput) :
        mGenerateOutput{generateOutput},
        mOutputSize3D{3, outputResolution.y, outputResolution.x}
    {
    }

    std::tuple<double, Array<float>> CvMatToOpOutput::format(const cv::Mat& cvInputData) const
    {
        try
        {
            // Security checks
            if (cvInputData.empty())
                error("Wrong input element (empty cvInputData).", __LINE__, __FUNCTION__, __FILE__);
            if (cvInputData.channels() != 3)
                error("Input images must be 3-channel BGR.", __LINE__, __FUNCTION__, __FILE__);
            // scaleInputToOutput - Scale between input and desired output size
            double scaleInputToOutput;
            Point<int> outputResolution;
            // Output = mOutputSize3D size
            if (mOutputSize3D[1] > 0 && mOutputSize3D[2] > 0)
            {
                outputResolution = Point<int>{mOutputSize3D[2], mOutputSize3D[1]};
                scaleInputToOutput = resizeGetScaleFactor(Point<int>{cvInputData.cols, cvInputData.rows},
                                                          outputResolution);
            }
            // Output = input size
            else
            {
                outputResolution = Point<int>{cvInputData.cols, cvInputData.rows};
                scaleInputToOutput = 1.;
            }
            // outputData - Reescale keeping aspect ratio and transform to float the output image
            Array<float> outputData;
            if (mGenerateOutput)
            {
                const cv::Mat frameWithOutputSize = resizeFixedAspectRatio(cvInputData, scaleInputToOutput,
                                                                           outputResolution);
                outputData.reset({3, outputResolution.y, outputResolution.x});
                uCharCvMatToFloatPtr(outputData.getPtr(), frameWithOutputSize, false);
            }
            // Return result
            return std::make_tuple(scaleInputToOutput, outputData);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_tuple(0., Array<float>{});
        }
    }
}
