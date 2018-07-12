#ifndef OPENPOSE_FACE_FACE_RENDERER_HPP
#define OPENPOSE_FACE_FACE_RENDERER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API FaceRenderer
    {
    public:
        virtual void initializationOnThread(){};

        void renderFace(Array<float>& outputData, const Array<float>& faceKeypoints,
                        const float scaleInputToOutput);

    private:
        virtual void renderFaceInherited(Array<float>& outputData, const Array<float>& faceKeypoints) = 0;
    };
}

#endif // OPENPOSE_FACE_FACE_RENDERER_HPP
