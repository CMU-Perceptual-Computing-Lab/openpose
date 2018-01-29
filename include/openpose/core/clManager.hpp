#ifndef OPENPOSE_CORE_CL_MANAGER_HPP
#define OPENPOSE_CORE_CL_MANAGER_HPP

#include <atomic>
#include <tuple>
#include <map>
#include <openpose/core/common.hpp>
#include <fstream>

#ifdef USE_OPENCL
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_MINIMUM_OPENCL_VERSION 120
    #define CL_HPP_TARGET_OPENCL_VERSION 120
    #include <CL/cl2.hpp>
    #include <viennacl/backend/opencl.hpp>
#endif

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
        std::map<std::string, cl::Program> clPrograms;
        std::map<std::string, cl::Kernel> clKernels;
        cl::Device device;
        cl::CommandQueue queue;
        cl::Context context;
        template <typename T> cl::Program buildProgramFromSource(std::string src, bool isFile = false);

    public:
        cl::Context& getContext();
        cl::CommandQueue& getQueue();
        cl::Device& getDevice();
        template <typename T> bool buildKernelIntoManager(std::string kernelName, std::string src = "", bool isFile = false);
        template <typename T> cl::Kernel& getKernelFromManager(std::string kernelName, std::string src = "", bool isFile = false);

    private:
        template <typename T> inline std::string getType()
        {
            std::string type = "";
            if ((std::is_same<T, float>::value))
                type = "float";
            else if ((std::is_same<T, double>::value))
                type = "double";
            else
                throw std::runtime_error("Error: Invalid CL type");
            return type;
        }

    public:
        template <typename T> static inline cl_buffer_region getBufferRegion(int origin, int size)
        {
            cl_buffer_region region;
            region.origin = sizeof(T) * origin;
            region.size = sizeof(T) * size;
            return region;
        }

        template <typename K, typename T> inline K getKernelFunctorFromManager(std::string kernelName, std::string src = "", bool isFile = false){
            return K(getKernelFromManager<T>(kernelName, src, isFile));
        }
    };
}

#endif // OPENPOSE_CORE_CL_MANAGER_HPP
