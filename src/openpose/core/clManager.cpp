#include <openpose/core/clManager.hpp>

namespace op
{
    std::shared_ptr<CLManager> CLManager::getInstance(int deviceId, int deviceType, bool getFromVienna)
    {
        static std::map<int, std::shared_ptr<CLManager>> clManagers;
        if(clManagers.count(deviceId))
            return clManagers[deviceId];
        else
        {
            clManagers[deviceId] = std::shared_ptr<CLManager>(new CLManager(deviceId, deviceType, getFromVienna));
            return clManagers[deviceId];
        }
    }

    CLManager::CLManager(int deviceId, int deviceType, bool getFromVienna)
    {
        if(getFromVienna)
        {
            context = cl::Context(viennacl::ocl::get_context(deviceId).handle().get(), true);
            queue = cl::CommandQueue(viennacl::ocl::get_context(deviceId).get_queue().handle().get(), true);
            op::log(std::to_string(context.getInfo<CL_CONTEXT_DEVICES>().size()));
            device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
            //context.printContext();
        }
        else
        {
            std::vector<cl::Platform> platforms;
            std::vector<cl::Device> devices;
            std::string deviceName;
            cl_uint type;
            try {
                cl::Platform::get(&platforms);
                switch(deviceType)
                {
                    case CL_DEVICE_TYPE_GPU:
                    {
                        type = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                        if( type == CL_SUCCESS)
                        {
                            // Get only relavent device
                            cl::Context allContext(devices);
                            std::vector<cl::Device> gpuDevices;
                            gpuDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                            bool deviceFound = false;
                            for(int i=0; i<gpuDevices.size(); i++){
                                if(i == deviceId){
                                    device = gpuDevices[i];
                                    context = cl::Context(device);
                                    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                                    deviceFound = true;
                                    op::log("Made new GPU Instance: " + std::to_string(deviceId));
                                    break;
                                }
                            }
                            if(!deviceFound)
                            {
                                throw std::runtime_error("Error: Invalid GPU ID");
                            }
                        }
                        else if(type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                        {
                            throw std::runtime_error("Error: GPU Invalid Device or Device not found");
                        }
                        break;
                    }

                    case CL_DEVICE_TYPE_CPU:
                    {
                        cl::Platform::get(&platforms);
                        type = platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
                        if( type == CL_SUCCESS)
                        {
                            // Get only relavent device
                            std::vector<cl::Device> devices;
                            cl::Context allContext(devices);
                            std::vector<cl::Device> cpuDevices;
                            cpuDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                            bool deviceFound = false;
                            for(int i=0; i<cpuDevices.size(); i++){
                                if(i == deviceId){
                                    device = cpuDevices[i];
                                    context = cl::Context(device);
                                    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                                    deviceFound = true;
                                    op::log("Made new CPU Instance: " + std::to_string(deviceId));
                                    break;
                                }
                            }
                            if(!deviceFound)
                            {
                                throw std::runtime_error("Error: Invalid CPU ID");
                            }
                        }
                        else if(type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                        {
                            throw std::runtime_error("Error: CPU Invalid Device or Device not found");
                        }
                        break;
                    }

                    case CL_DEVICE_TYPE_ACCELERATOR:
                    {
                        cl::Platform::get(&platforms);
                        type = platforms[0].getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
                        if( type == CL_SUCCESS)
                        {
                            // Get only relavent device
                            std::vector<cl::Device> devices;
                            cl::Context allContext(devices);
                            std::vector<cl::Device> accDevices;
                            accDevices = allContext.getInfo<CL_CONTEXT_DEVICES>();
                            bool deviceFound = false;
                            for(int i=0; i<accDevices.size(); i++){
                                if(i == deviceId){
                                    device = accDevices[i];
                                    context = cl::Context(device);
                                    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
                                    deviceFound = true;
                                    op::log("Made new ACC Instance: " + std::to_string(deviceId));
                                    break;
                                }
                            }
                            if(!deviceFound)
                            {
                                throw std::runtime_error("Error: Invalid ACC ID");
                            }
                        }
                        else if(type == CL_INVALID_DEVICE_TYPE || type == CL_DEVICE_NOT_FOUND)
                        {
                            throw std::runtime_error("Error: ACC Invalid Device or Device not found");
                        }
                        break;
                    }

                    default:
                    {
                        throw std::runtime_error("Error: No such CL Device Type");
                    }
                }
            }
            catch(cl::Error e) {
                op::log("Error: " + std::string(e.what()));
            }
        }
    }

    CLManager::~CLManager()
    {

    }

    cl::Context& CLManager::getContext()
    {
        return context;
    }

    cl::CommandQueue& CLManager::getQueue()
    {
        return queue;
    }

    cl::Device& CLManager::getDevice()
    {
        return device;
    }

    cl::Program CLManager::buildProgramFromSource(std::string src, bool isFile)
    {
        cl::Program program;
        try{
            if(isFile)
            {
                std::ifstream programFile((char*) src.c_str());
                std::string programString(std::istreambuf_iterator<char>(programFile),
                                                  (std::istreambuf_iterator<char>()));
                src = programString;
                //src = std::regex_replace(src, std::regex(";"), std::string(";\n"));
            }
            program = cl::Program(context, src, true);
        }
        catch(cl::BuildError e) {
            auto buildInfo = e.getBuildLog();
            for (auto &pair : buildInfo) {
                std::cerr << "Device: " << pair.first.getInfo<CL_DEVICE_NAME>() << std::endl << pair.second << std::endl << std::endl;
            }
            exit(-1);
        }
        return program;
    }

    bool CLManager::buildKernelIntoManager(std::string kernelName, std::string src, bool isFile){
        // Program not built
        if (!(clPrograms.find(src) != clPrograms.end()))
        {
            clPrograms[src] = buildProgramFromSource(src, isFile);
        }

        cl::Program& program = clPrograms[src];

        // Kernel not built
        if (!(clKernels.find(kernelName) != clKernels.end()))
        {
            clKernels[kernelName] = cl::Kernel(program, kernelName.c_str());
            op::log("Kernel " + kernelName + " built successfully");
            return true;
        }
        else
        {
            op::log("Kernel " + kernelName + " already built");
            return false;
        }
    }

    cl::Kernel& CLManager::getKernelFromManager(std::string kernelName){
        if (!(clKernels.find(kernelName) != clKernels.end()))
        {
            throw std::runtime_error("Error: Kernel not found in Manager: " + kernelName);
        }
        return clKernels[kernelName];
    }
}
