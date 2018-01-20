#include <openpose/core/clManager.hpp>
#include <iostream>
using namespace std;

namespace op
{
    CLManager& CLManager::getInstance()
    {
        static CLManager instance;
        return instance;
    }

    CLManager::CLManager(int deviceType)
    {
        std::vector<cl::Device> gpuDevices, cpuDevices, accDevices;
        std::string deviceName;
        cl_uint i, type;
        try {
            cl::Platform::get(&platforms);
            switch(deviceType)
            {
                case CL_DEVICE_TYPE_GPU:
                {
                    type = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
                    if( type == CL_SUCCESS)
                    {
                        //Create context and access device names
                        cl::Context ctx_(devices);
                        context = ctx_;
                        gpuDevices = context.getInfo<CL_CONTEXT_DEVICES>();
                        for(i=0; i<gpuDevices.size(); i++) {
                            deviceName = gpuDevices[i].getInfo<CL_DEVICE_NAME>();
                            queues.emplace_back(cl::CommandQueue(context, gpuDevices[i], CL_QUEUE_PROFILING_ENABLE));
                            op::log("Adding " + deviceName + " to queue");
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
                        // Create context and access device names
                        cl::Context ctx_(devices);
                        context = ctx_;
                        cpuDevices = context.getInfo<CL_CONTEXT_DEVICES>();
                        for(i=0; i<cpuDevices.size(); i++) {
                            deviceName = cpuDevices[i].getInfo<CL_DEVICE_NAME>();
                            queues.emplace_back(cl::CommandQueue(context, cpuDevices[i], CL_QUEUE_PROFILING_ENABLE));
                            op::log("Adding " + deviceName + " to queue");
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
                        // Create context and access device names
                        cl::Context ctx__(devices);
                        context = ctx__;
                        accDevices = context.getInfo<CL_CONTEXT_DEVICES>();
                        for(i=0; i<accDevices.size(); i++) {
                            deviceName = accDevices[i].getInfo<CL_DEVICE_NAME>();
                            queues.emplace_back(cl::CommandQueue(context, accDevices[i], CL_QUEUE_PROFILING_ENABLE));
                            op::log("Adding " + deviceName + " to queue");
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
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    CLManager::~CLManager()
    {

    }

}
