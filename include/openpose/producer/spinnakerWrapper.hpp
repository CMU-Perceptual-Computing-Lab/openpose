#ifndef OPENPOSE_PRODUCER_SPINNAKER_WRAPPER_HPP
#define OPENPOSE_PRODUCER_SPINNAKER_WRAPPER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    /**
     * SpinnakerWrapper is a subclass of SpinnakerWrapper. It decouples the final interface (meant to imitates
     * cv::VideoCapture) from the Spinnaker SDK wrapper.
     */
    class OP_API SpinnakerWrapper
    {
    public:
        /**
         * Constructor of SpinnakerWrapper. It opens all the available FLIR cameras
         * cameraIndex = -1 means that all cameras are taken
         */
        explicit SpinnakerWrapper(const std::string& cameraParameterPath, const Point<int>& cameraResolution,
                                  const bool undistortImage, const int cameraIndex = -1);

        ~SpinnakerWrapper();

        std::vector<cv::Mat> getRawFrames();

        std::vector<cv::Mat> getCameraMatrices() const;

        std::vector<cv::Mat> getCameraExtrinsics() const;

        std::vector<cv::Mat> getCameraIntrinsics() const;

        Point<int> getResolution() const;

        bool isOpened() const;

        void release();

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplSpinnakerWrapper;
        std::shared_ptr<ImplSpinnakerWrapper> upImpl;

        DELETE_COPY(SpinnakerWrapper);
    };
}

#endif // OPENPOSE_PRODUCER_SPINNAKER_WRAPPER_HPP
