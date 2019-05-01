#ifndef OPENPOSE_UTILITIES_PROFILER_HPP
#define OPENPOSE_UTILITIES_PROFILER_HPP

#include <chrono>
#include <string>
#include <openpose/core/macros.hpp>
#include <openpose/utilities/enumClasses.hpp>

namespace op
{
    // The following functions provides basic functions to measure time. Usage example:
    //     const auto timerInit = getTimerInit();
    //         // [Some code in here]
    //     const auto timeSeconds = getTimeSeconds(timerInit);
    //     const printTime(timeSeconds, "Function X took ", " seconds.");
    OP_API std::chrono::time_point<std::chrono::high_resolution_clock> getTimerInit();

    OP_API double getTimeSeconds(const std::chrono::time_point<std::chrono::high_resolution_clock>& timerInit);

    OP_API void printTime(
        const std::chrono::time_point<std::chrono::high_resolution_clock>& timerInit, const std::string& firstMessage,
        const std::string& secondMessage, const Priority priority);

    // The following functions will run REPS times and average the final time in seconds. Usage example:
    //     const auto REPS = 1000;
    //     double time = 0.;
    //     OP_PROFILE_INIT(REPS);
    //         // [Some code in here]
    //     OP_PROFILE_END(time, 1e3, REPS); // Time in msec. 1 = sec, 1e3 = msec, 1e6 = usec, 1e9 = nsec, etc.
    //     log("Function X took " + std::to_string(time) + " milliseconds.");
    #define OP_PROFILE_INIT(REPS) \
    { \
        const auto timerInit = getTimerInit(); \
        for (auto rep = 0 ; rep < (REPS) ; ++rep) \
        {
    #define OP_PROFILE_END(finalTime, factor, REPS) \
        } \
        (finalTime) = (factor)/(float)(REPS)*getTimeSeconds(timerInit); \
    }

    // The following functions will run REPS times, wait for the kernels to finish, and then average the final time
    // in seconds. Usage example:
    //     const auto REPS = 1000;
    //     double time = 0.;
    //     OP_CUDA_PROFILE_INIT(REPS);
    //         // [Some code with CUDA calls in here]
    //     OP_CUDA_PROFILE_END(time, 1e3, REPS); // Time in msec. 1 = sec, 1e3 = msec, 1e6 = usec, 1e9 = nsec, etc.
    //     log("Function X took " + std::to_string(time) + " milliseconds.");
    // Analogous to OP_PROFILE_INIT, but also waits for CUDA kernels to finish their asynchronous operations
    // It requires: #include <cuda_runtime.h>
    #define OP_CUDA_PROFILE_INIT(REPS) \
    { \
        cudaDeviceSynchronize(); \
        const auto timerInit = getTimerInit(); \
        for (auto rep = 0 ; rep < (REPS) ; ++rep) \
        {
    // Analogous to OP_PROFILE_END, but also waits for CUDA kernels to finish their asynchronous operations
    // It requires: #include <cuda_runtime.h>
    #define OP_CUDA_PROFILE_END(finalTime, factor, REPS) \
        } \
        cudaDeviceSynchronize(); \
        (finalTime) = (factor)/(float)(REPS)*getTimeSeconds(timerInit); \
        cudaCheck(__LINE__, __FUNCTION__, __FILE__); \
    }

    // Enable PROFILER_ENABLED on Makefile.config or CMake in order to use this function. Otherwise nothing will be outputted.
    // How to use - example:
    // For GPU - It can only be applied in the main.cpp file:
        // Profiler::profileGpuMemory(__LINE__, __FUNCTION__, __FILE__);
    // For time:
        // // ... inside continuous loop ...
        // const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
        // // functions to do...
        // Profiler::timerEnd(profilerKey);
        // Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__, NUMBER_ITERATIONS);
    class OP_API Profiler
    {
    public:
        static unsigned long long DEFAULT_X;

        // Non-thread safe, it must be performed at the beginning of the code before any parallelization occurs
        static void setDefaultX(const unsigned long long defaultX);

        static const std::string timerInit(const int line, const std::string& function, const std::string& file);

        static void timerEnd(const std::string& key);

        static void printAveragedTimeMsOnIterationX(
            const std::string& key, const int line, const std::string& function, const std::string& file,
            const unsigned long long x = DEFAULT_X);

        static void printAveragedTimeMsEveryXIterations(
            const std::string& key, const int line, const std::string& function, const std::string& file,
            const unsigned long long x = DEFAULT_X);

        static void profileGpuMemory(const int line, const std::string& function, const std::string& file);
    };
}

#endif // OPENPOSE_UTILITIES_PROFILER_HPP
