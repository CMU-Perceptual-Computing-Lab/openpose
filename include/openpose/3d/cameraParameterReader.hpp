#ifndef OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP
#define OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API CameraParameterReader
    {
    public:
        explicit CameraParameterReader();

        virtual ~CameraParameterReader();

        // cameraExtrinsics is optional
        explicit CameraParameterReader(const std::string& serialNumber,
                                       const Matrix& cameraIntrinsics,
                                       const Matrix& cameraDistortion,
                                       const Matrix& cameraExtrinsics = Matrix(),
                                       const Matrix& cameraExtrinsicsInitial = Matrix());

        // serialNumbers is optional. If empty, it will load all the XML files available in the
        // cameraParameterPath folder
        void readParameters(const std::string& cameraParameterPath,
                            const std::vector<std::string>& serialNumbers = {});

        // It simply calls the previous readParameters with a single element
        void readParameters(const std::string& cameraParameterPath,
                            const std::string& serialNumber);

        void writeParameters(const std::string& cameraParameterPath) const;

        unsigned long long getNumberCameras() const;

        const std::vector<std::string>& getCameraSerialNumbers() const;

        const std::vector<Matrix>& getCameraMatrices() const;

        const std::vector<Matrix>& getCameraDistortions() const;

        const std::vector<Matrix>& getCameraIntrinsics() const;

        const std::vector<Matrix>& getCameraExtrinsics() const;

        const std::vector<Matrix>& getCameraExtrinsicsInitial() const;

        bool getUndistortImage() const;

        void setUndistortImage(const bool undistortImage);

        void undistort(Matrix& frame, const unsigned int cameraIndex = 0u);

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplCameraParameterReader;
        std::shared_ptr<ImplCameraParameterReader> spImpl;

        DELETE_COPY(CameraParameterReader);
    };
}

#endif // OPENPOSE_3D_CAMERA_PARAMETER_READER_HPP
