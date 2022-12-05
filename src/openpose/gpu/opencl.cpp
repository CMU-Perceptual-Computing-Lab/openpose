#include <openpose_private/gpu/opencl.hcl> // Must be before below includes
#include <map>
#include <mutex>
#ifdef USE_OPENCL
    #include <openpose_private/gpu/cl2.hpp>
    #include <viennacl/backend/opencl.hpp>
    #include <caffe/caffe.hpp>
#endif

namespace op
{
    #ifdef USE_OPENCL
        void replaceAll(std::string &s, const std::string &search, const std::string &replace)
        {
            for (size_t pos = 0; ; pos += replace.length())
            {
                // Locate the substring to replace
                pos = s.find( search, pos );
                if ( pos == std::string::npos ) break;
                // Replace by erasing and inserting
                s.erase( pos, search.length() );
                s.insert( pos, replace );
            }
        }

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

        template <typename T>
        bool buildProgramFromSource(cl::Program& program, cl::Context& context, std::string src, bool isFile = false)
        {
            #ifdef USE_OPENCL
                try
                {
                    std::string type = getType<T>();
                    if (isFile)
                    {
                        std::ifstream programFile((char*) src.c_str());
                        std::string programString(std::istreambuf_iterator<char>(programFile),
                                                          (std::istreambuf_iterator<char>()));
                        src = programString;
                    }
                    //src = std::regex_replace(src, std::regex("Type"), std::string(type));
                    replaceAll(src, "Type", type);
                    program = cl::Program(context, src, true);
                }
                #if defined(USE_OPENCL) && defined(CL_HPP_ENABLE_EXCEPTIONS)
                catch (cl::BuildError e)
                {
                    auto buildInfo = e.getBuildLog();
                    for (auto &pair : buildInfo)
                        std::cerr << "Device: " << pair.first.getInfo<CL_DEVICE_NAME>() << std::endl <<
                                     pair.second << std::endl;
                        error("OpenCL error: OpenPose crashed due to the previously printed errors.",
                              __LINE__, __FUNCTION__, __FILE__);
                }
                #endif
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
                return true;
            #else
                UNUSED(program);
                UNUSED(src);
                UNUSED(isFile);
                error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
                return false;
            #endif
        }
    #endif

    std::shared_ptr<OpenCL> OpenCL::getInstance(const int deviceId, const int deviceType, bool getFromVienna)
    {
        static std::mutex managerMutex;
        static std::map<int, std::shared_ptr<OpenCL>> clManagers;
        if (clManagers.count(deviceId))
            return clManagers[deviceId];
        else
        {
            std::unique_lock<std::mutex> lock{managerMutex};
            clManagers[deviceId] = std::shared_ptr<OpenCL>(new OpenCL(deviceId, deviceType, getFromVienna));
            lock.unlock();
            return clManagers[deviceId];
        }
    }

    struct OpenCL::ImplCLManager
    {
    public:
        #ifdef USE_OPENCL
            std::map<std::string, cl::Program> mClPrograms;
            std::map<std::string, cl::Kernel> mClKernels;
            int mId;
            cl::Device mDevice;
            cl::CommandQueue mQueue;
            cl::Context mContext;
        #endif

        ImplCLManager()
        {
        }
    };

    OpenCL::OpenCL(const int deviceId, const int deviceType, bool getFromVienna)
        : upImpl{new ImplCLManager{}}
    {
        #ifdef USE_OPENCL
            upImpl->mId = deviceId;
            if (getFromVienna)
            {
                upImpl->mContext = cl::Context(caffe::Caffe::GetOpenCLContext(deviceId, 0), true);
                upImpl->mQueue = cl::CommandQueue(caffe::Caffe::GetOpenCLQueue(deviceId, 0), true);
                upImpl->mDevice = upImpl->mContext.getInfo<CL_CONTEXT_DEVICES>()[0];
                //context = cl::Context(viennacl::ocl::get_context(deviceId).handle().get(), true);
                //queue = cl::CommandQueue(viennacl::ocl::get_context(deviceId).get_queue().handle().get(), true);
            }
            else
            {
                std::vector<cl::Platform> platforms;
                std::vector<cl::Device> devices;
                std::string deviceName;
                //cl_uint type;
                try
                {
                    cl::Platform::get(&platforms);
                    switch (deviceType)
                    {
                        case CL_DEVICE_TYPE_GPU:
                        {
                            auto type = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                            if ( type == CL_SUCCESS)
                            {
                                // Get only relevant device
                                cl::Context allContext(devices);
                                std::vector<cl::Device> gpuDevices;
                                gpuDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                                bool deviceFound = false;
                                for (size_t i=0; i<gpuDevices.size(); i++)
                                {
                                    if (i == (unsigned int)deviceId)
                                    {
                                        upImpl->mDevice = gpuDevices[i];
                                        upImpl->mContext = cl::Context(upImpl->mDevice);
                                        upImpl->mQueue = cl::CommandQueue(upImpl->mContext, upImpl->mDevice,
                                                                          CL_QUEUE_PROFILING_ENABLE);
                                        deviceFound = true;
                                        opLog("Made new GPU Instance: " + std::to_string(deviceId));
                                        break;
                                    }
                                }
                                if (!deviceFound)
                                    throw std::runtime_error("Error: Invalid GPU ID");
                            }
                            else if (type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                                throw std::runtime_error("Error: GPU Invalid Device or Device not found");
                            break;
                        }

                        case CL_DEVICE_TYPE_CPU:
                        {
                            cl::Platform::get(&platforms);
                            auto type = platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
                            if ( type == CL_SUCCESS)
                            {
                                // Get only relevant device
                                std::vector<cl::Device> devices;
                                cl::Context allContext(devices);
                                std::vector<cl::Device> cpuDevices;
                                cpuDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                                bool deviceFound = false;
                                for (size_t i=0; i<cpuDevices.size(); i++){
                                    if (i == (unsigned int)deviceId){
                                        upImpl->mDevice = cpuDevices[i];
                                        upImpl->mContext = cl::Context(upImpl->mDevice);
                                        upImpl->mQueue = cl::CommandQueue(upImpl->mContext, upImpl->mDevice,
                                                                          CL_QUEUE_PROFILING_ENABLE);
                                        deviceFound = true;
                                        opLog("Made new CPU Instance: " + std::to_string(deviceId));
                                        break;
                                    }
                                }
                                if (!deviceFound)
                                    throw std::runtime_error("Error: Invalid CPU ID");
                            }
                            else if (type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                                throw std::runtime_error("Error: CPU Invalid Device or Device not found");
                            break;
                        }

                        case CL_DEVICE_TYPE_ACCELERATOR:
                        {
                            cl::Platform::get(&platforms);
                            auto type = platforms[0].getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
                            if ( type == CL_SUCCESS)
                            {
                                // Get only relevant device
                                std::vector<cl::Device> devices;
                                cl::Context allContext(devices);
                                std::vector<cl::Device> accDevices;
                                accDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                                bool deviceFound = false;
                                for (size_t i=0; i<accDevices.size(); i++)
                                {
                                    if (i == (unsigned int)deviceId)
                                    {
                                        upImpl->mDevice = accDevices[i];
                                        upImpl->mContext = cl::Context(upImpl->mDevice);
                                        upImpl->mQueue = cl::CommandQueue(upImpl->mContext, upImpl->mDevice,
                                                                          CL_QUEUE_PROFILING_ENABLE);
                                        deviceFound = true;
                                        opLog("Made new ACC Instance: " + std::to_string(deviceId));
                                        break;
                                    }
                                }
                                if (!deviceFound)
                                    throw std::runtime_error("Error: Invalid ACC ID");
                            }
                            else if (type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                                throw std::runtime_error("Error: ACC Invalid Device or Device not found");
                            break;
                        }

                        default:
                        {
                            throw std::runtime_error("Error: No such CL Device Type");
                        }
                    }
                }
                #if defined(USE_OPENCL) && defined(CL_HPP_ENABLE_EXCEPTIONS)
                catch (cl::Error e)
                {
                    opLog("Error: " + std::string(e.what()));
                }
                #endif
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }
        #else
            UNUSED(deviceId);
            UNUSED(deviceType);
            UNUSED(getFromVienna);
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    OpenCL::~OpenCL()
    {
    }

    cl::CommandQueue& OpenCL::getQueue()
    {
        #ifdef USE_OPENCL
            return upImpl->mQueue;
        #else
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
            throw std::runtime_error("");
        #endif
    }

    cl::Device& OpenCL::getDevice()
    {
        #ifdef USE_OPENCL
            return upImpl->mDevice;
        #else
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
            throw std::runtime_error("");
        #endif
    }

    cl::Context& OpenCL::getContext()
    {
        #ifdef USE_OPENCL
            return upImpl->mContext;
        #else
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
            throw std::runtime_error("");
        #endif
    }

    template <typename T>
    bool OpenCL::buildKernelIntoManager(const std::string& kernelName, const std::string& src, bool isFile)
    {
        #ifdef USE_OPENCL
            // Set type
            std::string type = getType<T>();
            std::string key = kernelName + "_" + type;

            // Program not built
            if (!(upImpl->mClPrograms.find(key) != upImpl->mClPrograms.end()))
            {
                cl::Program program;
                buildProgramFromSource<T>(program, upImpl->mContext, src, isFile);
                upImpl->mClPrograms[key] = program;
            }

            cl::Program& program = upImpl->mClPrograms[key];

            // Kernel not built
            if (!(upImpl->mClKernels.find(key) != upImpl->mClKernels.end()))
            {
                upImpl->mClKernels[key] = cl::Kernel(program, kernelName.c_str());
                opLog("Kernel: " + kernelName + " Type: " + type + + " GPU: " + std::to_string(upImpl->mId) +
                    " built successfully");
                return true;
            }
            else
            {
                opLog("Kernel " + kernelName + " already built");
                return false;
            }
        #else
            UNUSED(kernelName);
            UNUSED(src);
            UNUSED(isFile);
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
            return false;
        #endif
    }

    template <typename T>
    cl::Kernel& OpenCL::getKernelFromManager(const std::string& kernelName, const std::string& src, bool isFile)
    {
        #ifdef USE_OPENCL
        // Set type
        std::string type = getType<T>();
        std::string key = kernelName + "_" + type;

        if (!(upImpl->mClKernels.find(key) != upImpl->mClKernels.end()))
        {
            if (!src.size())
                throw std::runtime_error("Error: Kernel " + kernelName + " Type: " + type + " not found in Manager");
            else
            {
                buildKernelIntoManager<T>(kernelName, src, isFile);
            }
        }
        return upImpl->mClKernels[key];
        #else
        UNUSED(kernelName);
        UNUSED(src);
        UNUSED(isFile);
        error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
              " functionality.", __LINE__, __FUNCTION__, __FILE__);
        throw std::runtime_error("");
        #endif
    }

    std::string OpenCL::clErrorToString(int err)
    {
        #ifdef USE_OPENCL
            switch (err)
            {
            case CL_SUCCESS: return "Success";
            case CL_DEVICE_NOT_FOUND: return "Device Not Found";
            case CL_DEVICE_NOT_AVAILABLE: return "Device Not Available";
            case CL_COMPILER_NOT_AVAILABLE: return "Compiler Not Available";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory Object Allocation Failure";
            case CL_OUT_OF_RESOURCES: return "Out of Resources";
            case CL_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
            case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling Information Not Available";
            case CL_MEM_COPY_OVERLAP: return "Memory Copy Overlap";
            case CL_IMAGE_FORMAT_MISMATCH: return "Image Format Mismatch";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image Format Not Supported";
            case CL_BUILD_PROGRAM_FAILURE: return "Build Program Failure";
            case CL_MAP_FAILURE: return "Map Failure";
            case CL_INVALID_VALUE: return "Invalid Value";
            case CL_INVALID_DEVICE_TYPE: return "Invalid Device Type";
            case CL_INVALID_PLATFORM: return "Invalid Platform";
            case CL_INVALID_DEVICE: return "Invalid Device";
            case CL_INVALID_CONTEXT: return "Invalid Context";
            case CL_INVALID_QUEUE_PROPERTIES: return "Invalid Queue Properties";
            case CL_INVALID_COMMAND_QUEUE: return "Invalid Command Queue";
            case CL_INVALID_HOST_PTR: return "Invalid Host Pointer";
            case CL_INVALID_MEM_OBJECT: return "Invalid Memory Object";
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid Image Format Descriptor";
            case CL_INVALID_IMAGE_SIZE: return "Invalid Image Size";
            case CL_INVALID_SAMPLER: return "Invalid Sampler";
            case CL_INVALID_BINARY: return "Invalid Binary";
            case CL_INVALID_BUILD_OPTIONS: return "Invalid Build Options";
            case CL_INVALID_PROGRAM: return "Invalid Program";
            case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid Program Executable";
            case CL_INVALID_KERNEL_NAME: return "Invalid Kernel Name";
            case CL_INVALID_KERNEL_DEFINITION: return "Invalid Kernel Definition";
            case CL_INVALID_KERNEL: return "Invalid Kernel";
            case CL_INVALID_ARG_INDEX: return "Invalid Argument Index";
            case CL_INVALID_ARG_VALUE: return "Invalid Argument Value";
            case CL_INVALID_ARG_SIZE: return "Invalid Argument Size";
            case CL_INVALID_KERNEL_ARGS: return "Invalid Kernel Arguments";
            case CL_INVALID_WORK_DIMENSION: return "Invalid Work Dimension";
            case CL_INVALID_WORK_GROUP_SIZE: return "Invalid Work Group Size";
            case CL_INVALID_WORK_ITEM_SIZE: return "Invalid Work Item Size";
            case CL_INVALID_GLOBAL_OFFSET: return "Invalid Global Offset";
            case CL_INVALID_EVENT_WAIT_LIST: return "Invalid Event Wait List";
            case CL_INVALID_EVENT: return "Invalid Event";
            case CL_INVALID_OPERATION: return "Invalid Operation";
            case CL_INVALID_GL_OBJECT: return "Invalid GL Object";
            case CL_INVALID_BUFFER_SIZE: return "Invalid Buffer Size";
            case CL_INVALID_MIP_LEVEL: return "Invalid MIP Level";
            case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid Global Work Size";
            #ifdef CL_VERSION_1_2
            case CL_COMPILE_PROGRAM_FAILURE: return "Compile Program Failure";
            case CL_LINKER_NOT_AVAILABLE: return "Linker Not Available";
            case CL_LINK_PROGRAM_FAILURE: return "Link Program Failure";
            case CL_DEVICE_PARTITION_FAILED: return "Device Partition Failed";
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "Kernel Argument Info Not Available";
            case CL_INVALID_PROPERTY: return "Invalid Property";
            case CL_INVALID_IMAGE_DESCRIPTOR: return "Invalid Image Descriptor";
            case CL_INVALID_COMPILER_OPTIONS: return "Invalid Compiler Options";
            case CL_INVALID_LINKER_OPTIONS: return "Invalid Linker Options";
            case CL_INVALID_DEVICE_PARTITION_COUNT: return "Invalid Device Partition Count";
            #endif // CL_VERSION_1_2
            #ifdef CL_VERSION_2_0
            case CL_INVALID_PIPE_SIZE: return "Invalid Pipe Size";
            case CL_INVALID_DEVICE_QUEUE: return "Invalid Device Queue";
            #endif
            default: {
                std::stringstream s;
                s << "Unknown OpenCL Error (" << err << ")";
                return s.str();
            }
            }
        #else
            UNUSED(err);
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
            return "";
        #endif
    }

    int OpenCL::getTotalGPU()
    {
        #ifdef USE_OPENCL
            std::vector<cl::Platform> platforms;
            std::vector<cl::Device> devices;
            cl_uint type;
            try
            {
                cl::Platform::get(&platforms);
                if (!platforms.size())
                    return -1;

                // Special Case for Apple which has CPU OpenCL Device too
                int cpu_device_count = 0;
                #ifdef __APPLE__
                    type = platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
                    if (type == CL_SUCCESS)
                        cpu_device_count = devices.size();
                #endif

                type = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                if (type == CL_SUCCESS)
                    return devices.size() + cpu_device_count;
                else
                {
                    error("No GPU Devices were found. OpenPose only supports GPU OpenCL", __LINE__, __FUNCTION__, __FILE__);
                    return -1;
                }
            }
            #if defined(USE_OPENCL) && defined(CL_HPP_ENABLE_EXCEPTIONS)
            catch (cl::Error& e)
            {
                opLog("Error: " + std::string(e.what()));
            }
            #endif
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #else
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
        #endif
        return -1;
    }

    template <typename T> void OpenCL::getBufferRegion(cl_buffer_region& region, const int origin, const int size)
    {
        #ifdef USE_OPENCL
            region.origin = sizeof(T) * origin;
            region.size = sizeof(T) * size;
        #else
            UNUSED(origin);
            UNUSED(size);
            UNUSED(region);
            error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
                  " functionality.", __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    int OpenCL::getAlignment()
    {
        #ifdef USE_OPENCL
        cl::Device& device = this->getDevice();
        cl_uint mem_align;
        clGetDeviceInfo(device.get(), CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(mem_align), &mem_align, nullptr);
        return mem_align;
        #else
        error("OpenPose must be compiled with the `USE_OPENCL` macro definition in order to use this"
              " functionality.", __LINE__, __FUNCTION__, __FILE__);
        return 0;
        #endif
    }

    template void OpenCL::getBufferRegion<float>(cl_buffer_region& region, const int origin, const int size);
    template void OpenCL::getBufferRegion<double>(cl_buffer_region& region, const int origin, const int size);
    template cl::Kernel&  OpenCL::getKernelFromManager<float>(const std::string& kernelName, const std::string& src, bool isFile);
    template cl::Kernel& OpenCL::getKernelFromManager<double>(const std::string& kernelName, const std::string& src, bool isFile);
    template bool OpenCL::buildKernelIntoManager<float>(const std::string& kernelName, const std::string& src, bool isFile);
    template bool OpenCL::buildKernelIntoManager<double>(const std::string& kernelName, const std::string& src, bool isFile);
}
