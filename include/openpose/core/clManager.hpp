#ifndef OPENPOSE_CORE_CL_MANAGER_HPP
#define OPENPOSE_CORE_CL_MANAGER_HPP

#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <openpose/core/common.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 200 // Need to set to 120 on CUDA 8
#define CL_HPP_TARGET_OPENCL_VERSION 200 // Need to set to 120 on CUDA 8

// Singleton structure
// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern

namespace op
{
    class CLManager
    {
    private:
    public:
        static std::shared_ptr<CLManager> getInstance(int deviceId = 0, int deviceType = CL_DEVICE_TYPE_GPU, bool getFromVienna = false);
        ~CLManager();
        DELETE_COPY(CLManager);

    private:
        CLManager(int deviceId, int deviceType, bool getFromVienna);
        struct ImplCLManager;
        std::unique_ptr<ImplCLManager> upImpl;

    public:
        cl::Context& getContext();
        cl::CommandQueue& getQueue();
        cl::Device& getDevice();
        template <typename T> bool buildKernelIntoManager(std::string kernelName, std::string src = "", bool isFile = false);
        template <typename T> cl::Kernel& getKernelFromManager(std::string kernelName, std::string src = "", bool isFile = false);
        template <typename K, typename T> inline K getKernelFunctorFromManager(std::string kernelName, std::string src = "", bool isFile = false){            
            return K(getKernelFromManager<T>(kernelName, src, isFile));
        }
    public:       
        template <typename T> static void getBufferRegion(cl_buffer_region& region, int origin, int size);
        static std::string clErrorToString(int err);
        static int getTotalGPU();
    };
}

#endif // OPENPOSE_CORE_CL_MANAGER_HPP
