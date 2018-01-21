#ifndef OPENPOSE_CORE_CL_MANAGER_HPP
#define OPENPOSE_CORE_CL_MANAGER_HPP

#include <atomic>
#include <tuple>
#include <map>
#include <openpose/core/common.hpp>
#include <fstream>

#ifdef USE_OPENCL
    #define __CL_ENABLE_EXCEPTIONS
    #include <CL/cl.hpp>
#endif

// Singleton structure
// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern

namespace op
{
    class OP_API CLManager
    {
    public:
        static CLManager& getInstance();

    private:
        explicit CLManager(int deviceType = CL_DEVICE_TYPE_GPU);
        ~CLManager();
        DELETE_COPY(CLManager);

    private:
        std::map<std::string, cl::Program> clPrograms;
        std::map<std::string, cl::Kernel> clKernels;
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        std::vector<cl::CommandQueue> queues;
        cl::Context context;
        cl::Program buildProgramFromSource(std::string src, bool isFile = false);

    public:
        cl::Context& getContext();
        cl::CommandQueue& getQueue(size_t gpuID = 0);
        cl::Device& getDevice(size_t gpuID = 0);
        bool buildKernelIntoManager(std::string kernelName, std::string src, bool isFile = false);
        cl::Kernel& getKernelFromManager(std::string kernelName);

    };
}

#endif // OPENPOSE_CORE_CL_MANAGER_HPP
