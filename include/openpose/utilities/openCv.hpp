#ifndef OPENPOSE_UTILITIES_OPEN_CV_HPP
#define OPENPOSE_UTILITIES_OPEN_CV_HPP

#include <openpose/core/common.hpp>

namespace op
{
    OP_API void unrollArrayToUCharCvMat(Matrix& matResult, const Array<float>& array);

    OP_API void uCharCvMatToFloatPtr(float* floatPtrImage, const Matrix& matImage, const int normalize);

    OP_API double resizeGetScaleFactor(const Point<int>& initialSize, const Point<int>& targetSize);

    OP_API void keepRoiInside(Rectangle<int>& roi, const int imageWidth, const int imageHeight);

    /**
     * It performs rotation and flipping over the desired Mat.
     * @param cvMat Mat with the frame matrix to be rotated and/or flipped.
     * @param rotationAngle How much the cvMat element should be rotated. 0 would mean no rotation.
     * @param flipFrame Whether to flip the cvMat element. Set to false to disable it.
     */
    OP_API void rotateAndFlipFrame(Matrix& cvMat, const double rotationAngle, const bool flipFrame = false);

    /**
     * Wrapper of CV_CAP_PROP_FRAME_COUNT to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvCapPropFrameCount();

    /**
     * Wrapper of CV_CAP_PROP_FRAME_FPS to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvCapPropFrameFps();

    /**
     * Wrapper of CV_CAP_PROP_FRAME_WIDTH to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvCapPropFrameWidth();

    /**
     * Wrapper of CV_CAP_PROP_FRAME_HEIGHT to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvCapPropFrameHeight();

    /**
     * Wrapper of CV_FOURCC to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvFourcc(const char c1, const char c2, const char c3, const char c4);

    /**
     * Wrapper of CV_IMWRITE_JPEG_QUALITY to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvImwriteJpegQuality();

    /**
     * Wrapper of CV_IMWRITE_PNG_COMPRESSION to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvImwritePngCompression();

    /**
     * Wrapper of CV_LOAD_IMAGE_ANYDEPTH to avoid leaving OpenCV dependencies on headers.
     */
    OP_API int getCvLoadImageAnydepth();
}

#endif // OPENPOSE_UTILITIES_OPEN_CV_HPP
