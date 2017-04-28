#ifndef OPENPOSE__WRAPPER__WRAPPER_STRUCT_FACE_HPP
#define OPENPOSE__WRAPPER__WRAPPER_STRUCT_FACE_HPP

namespace op
{
    namespace experimental
    {
        /**
         * WrapperStructFace: Face estimation and rendering configuration struct.
         * DO NOT USE. CODE TO BE FINISHED.
         * WrapperStructFace allows the user to set up the face estimation and rendering parameters that will be used for the OpenPose Wrapper
         * class.
         */
        struct WrapperStructFace
        {
            /**
             * PROVISIONAL PARAMETER. IT WILL BE CHANGED.
             * Whether to extract and render face.
             */
            bool extractAndRenderFace;

            /**
             * Constructor of the struct.
             * It has the recommended and default values we recommend for each element of the struct.
             * Since all the elements of the struct are public, they can also be manually filled.
             */
            WrapperStructFace(const bool extractAndRenderFace = false);
        };
    }
}

#endif // OPENPOSE__WRAPPER__WRAPPER_STRUCT_FACE_HPP
