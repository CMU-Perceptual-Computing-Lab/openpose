#ifndef OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP
#define OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP

#include <openpose/wrapper/wrapperStructFace.hpp>
#include <openpose/wrapper/wrapperStructHand.hpp>
#include <openpose/wrapper/wrapperStructInput.hpp>
#include <openpose/wrapper/wrapperStructOutput.hpp>
#include <openpose/wrapper/wrapperStructPose.hpp>

namespace op
{
    /**
     * It checks that no wrong/contradictory flags are enabled for Wrapper
     * @param wrapperStructPose
     * @param wrapperStructFace
     * @param wrapperStructHand
     * @param wrapperStructInput
     * @param wrapperStructOutput
     * @param renderOutput
     * @param userOutputWsEmpty
     * @param threadManagerMode
     */
    OP_API void wrapperConfigureSecurityChecks(const WrapperStructPose& wrapperStructPose,
                                               const WrapperStructFace& wrapperStructFace,
                                               const WrapperStructHand& wrapperStructHand,
                                               const WrapperStructInput& wrapperStructInput,
                                               const WrapperStructOutput& wrapperStructOutput,
                                               const bool renderOutput,
                                               const bool userOutputWsEmpty,
                                               const ThreadManagerMode threadManagerMode);
}

#endif // OPENPOSE_WRAPPER_WRAPPER_AUXILIARY_HPP
