#ifndef OPENPOSE__WRAPPER__WRAPPER_STRUCT_HANDS_HPP
#define OPENPOSE__WRAPPER__WRAPPER_STRUCT_HANDS_HPP

namespace op
{
    namespace experimental
    {
        /**
         * WrapperStructHands: Hands estimation and rendering configuration struct.
         * DO NOT USE. CODE TO BE FINISHED.
         * WrapperStructHands allows the user to set up the hands estimation and rendering parameters that will be used for the OpenPose Wrapper
         * class.
         */
        struct WrapperStructHands
        {
            /**
             * PROVISIONAL PARAMETER. IT WILL BE CHANGED.
             * Whether to extract and render hands.
             */
            bool extractAndRenderHands;

            /**
             * Constructor of the struct.
             * It has the recommended and default values we recommend for each element of the struct.
             * Since all the elements of the struct are public, they can also be manually filled.
             */
            WrapperStructHands(const bool extractAndRenderHands = false);
        };
    }
}

#endif // OPENPOSE__WRAPPER__WRAPPER_STRUCT_HANDS_HPP
