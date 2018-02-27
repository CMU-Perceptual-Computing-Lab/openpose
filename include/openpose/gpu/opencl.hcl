#ifndef OPENPOSE_CORE_OPENCL_HPP
#define OPENPOSE_CORE_OPENCL_HPP

#include <openpose/core/common.hpp>

#define MULTI_LINE_STRING(ARG) #ARG

#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef LOWER_CL_VERSION
    #define CL_HPP_MINIMUM_OPENCL_VERSION 120
    #define CL_HPP_TARGET_OPENCL_VERSION 120
#else
    #define CL_HPP_MINIMUM_OPENCL_VERSION 200
    #define CL_HPP_TARGET_OPENCL_VERSION 200
#endif

typedef struct _cl_buffer_region cl_buffer_region;
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
namespace cl
{
    class CommandQueue;
    class Kernel;
}

// Singleton structure
// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern

namespace op
{
    class OP_API OpenCL
    {
    public:
        static std::shared_ptr<OpenCL> getInstance(const int deviceId = 0, const int deviceType = CL_DEVICE_TYPE_GPU,
                                                   bool getFromVienna = false);
        ~OpenCL();

        cl::CommandQueue& getQueue();

        template <typename T>
        bool buildKernelIntoManager(const std::string& kernelName, const std::string& src = "", bool isFile = false);

        template <typename T>
        cl::Kernel& getKernelFromManager(const std::string& kernelName, const std::string& src = "", bool isFile = false);

        template <typename K, typename T>
        inline K getKernelFunctorFromManager(const std::string& kernelName, const std::string& src = "", bool isFile = false)
        {            
            return K(getKernelFromManager<T>(kernelName, src, isFile));
        }
      
        template <typename T> static void getBufferRegion(cl_buffer_region& region, const int origin, const int size);

        static std::string clErrorToString(int err);

        static int getTotalGPU();

    private:
        struct ImplCLManager;
        std::unique_ptr<ImplCLManager> upImpl;

        OpenCL(const int deviceId, const int deviceType, bool getFromVienna);

        DELETE_COPY(OpenCL);
    };
}

#endif // OPENPOSE_CORE_OPENCL_HPP
