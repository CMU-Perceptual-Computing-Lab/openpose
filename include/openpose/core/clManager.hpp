#ifndef OPENPOSE_CORE_CL_MANAGER_HPP
#define OPENPOSE_CORE_CL_MANAGER_HPP

#include <atomic>
#include <tuple>
#include <map>
#include <openpose/core/common.hpp>

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
        explicit CLManager(int deviceType = CL_DEVICE_TYPE_GPU);
        ~CLManager();
        DELETE_COPY(CLManager);
        static CLManager& getInstance();

    private:


    private:
        std::map<std::string, cl::Program> clPrograms;
        std::map<std::string, cl::Kernel> clKernels;
        std::string clPath;
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        std::vector<cl::CommandQueue> queues;
        cl::Context context;
        void initCL();
        void createDevicesAndContext();

    public:
        void setCLPath(std::string path);
        void destroyInstance();
        cl::Context getContext();
        cl::CommandQueue getQueue();
        cl::Program buildProgramFromSource(std::string filename);
        cl::Program buildProgramFromBinary(std::string filename);
        void saveBinary(cl::Program* program, std::string filename);
        cl::Device getDevice();

    };
}

#endif // OPENPOSE_CORE_CL_MANAGER_HPP
