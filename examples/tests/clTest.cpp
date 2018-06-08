// ------------------------- OpenPose Resize Layer Testing -------------------------

#include <openpose/headers.hpp>
#ifdef USE_OPENCL
#include <openpose/gpu/opencl.hcl>
#include <openpose/gpu/cl2.hpp>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif
#ifdef USE_CAFFE
#include <caffe/net.hpp>
#endif

DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");

//    cv::Mat gpuResize(cv::Mat& img, const cv::Size& newSize)
//    {
//        #ifdef USE_CUDA
//            // Upload to Source to GPU
//            float* cpuPtr = &img.at<float>(0);
//            float* gpuPtr;
//            cudaMallocHost((void **)&gpuPtr, img.size().width * img.size().height * sizeof(float));
//            cudaMemcpy(gpuPtr, cpuPtr, img.size().width * img.size().height * sizeof(float),
//                       cudaMemcpyHostToDevice);

//            // Upload to Dest to GPU
//            cv::Mat newImg = cv::Mat(newSize,CV_32FC1,cv::Scalar(0));
//            float* newCpuPtr = &newImg.at<float>(0);
//            float* newGpuPtr;
//            cudaMallocHost((void **)&newGpuPtr, newSize.width * newSize.height * sizeof(float));
//            cudaMemcpy(newGpuPtr, newCpuPtr, newSize.width * newSize.height * sizeof(float),
//                       cudaMemcpyHostToDevice);

//            std::vector<const float*> sourcePtrs;
//            sourcePtrs.emplace_back(gpuPtr);
//            std::array<int, 4> targetSize = {1,1,newImg.size().height,newImg.size().width};
//            std::array<int, 4> sourceSize = {1,1,img.size().height,img.size().width};
//            std::vector<std::array<int, 4>> sourceSizes;
//            sourceSizes.emplace_back(sourceSize);
//            op::resizeAndMergeGpu(newGpuPtr, sourcePtrs, targetSize, sourceSizes);
//            cudaMemcpy(newCpuPtr, newGpuPtr, newImg.size().width * newImg.size().height * sizeof(float),
//                       cudaMemcpyDeviceToHost);

//            cudaFree(gpuPtr);
//            cudaFree(newGpuPtr);
//            return newImg;
//        #else
//            UNUSED(img);
//            UNUSED(newSize);
//            op::error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
//                  " this functionality.", __LINE__, __FUNCTION__, __FILE__);
//        #endif
//    }

//    cv::Mat cpuResize(cv::Mat& img, cv::Size newSize)
//    {
//        // Upload to Source to GPU
//        float* cpuPtr = &img.at<float>(0);

//        // Upload to Dest to GPU
//        cv::Mat newImg = cv::Mat(newSize,CV_32FC1,cv::Scalar(0));

//        std::vector<const float*> sourcePtrs;
//        sourcePtrs.emplace_back(cpuPtr);
//        std::array<int, 4> targetSize = {1,1,newImg.size().height,newImg.size().width};
//        std::array<int, 4> sourceSize = {1,1,img.size().height,img.size().width};
//        std::vector<std::array<int, 4>> sourceSizes;
//        sourceSizes.emplace_back(sourceSize);
//        op::resizeAndMergeCpu(&newImg.at<float>(0), sourcePtrs, targetSize, sourceSizes);

//        return newImg;
//    }

typedef cl::KernelFunctor<cl::Buffer, int, int, float> ScaleFunctor;
const std::string scaleKernelString = MULTI_LINE_STRING(
    __kernel void scaleKernel(__global float* targetPtr, const int targetWidth, const int targetHeight, const float scale)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int c = get_global_id(2);

        __global float* targetPtrC = &targetPtr[c*targetWidth*targetHeight];
        targetPtrC[y*targetWidth+x] *= scale;
    }
);

int clTest()
{
    try
    {
        // logging_level
        cv::Mat img = op::loadImage(FLAGS_image_path, CV_LOAD_IMAGE_GRAYSCALE);
        if(img.empty())
            op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        cv::Mat imgResize; cv::resize(img, imgResize, cv::Size(368,368));
        cv::Mat imgFloat; imgResize.convertTo(imgFloat, CV_32FC1);
        imgFloat /= 255.;
        int imageVolume = imgResize.size().width * imgResize.size().height;

        // Setup caffe
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        std::vector<int> devices;
        const int maxNumberGpu = op::OpenCL::getTotalGPU();
        for (auto i = 0; i < maxNumberGpu; i++){
            devices.emplace_back(i);
            std::cout << i << std::endl;
        }
        caffe::Caffe::SetDevices(devices);

        // Load model
        std::unique_ptr<caffe::Net<float>> upCaffeNet;
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SelectDevice(0, true);
        upCaffeNet.reset(new caffe::Net<float>{"models/pose/coco/pose_deploy_linevec.prototxt", caffe::TEST, caffe::Caffe::GetDefaultDevice()});
        upCaffeNet->CopyTrainedLayersFrom("models/pose/coco/pose_iter_440000.caffemodel");
        op::OpenCL::getInstance(0, CL_DEVICE_TYPE_GPU, true);

        // Reshape net to image size
        upCaffeNet->blobs()[0]->Reshape({1,3,imgResize.size().width,imgResize.size().height});
        upCaffeNet->Reshape();

        // GPU Test
        cv::Mat finalImage;
        try{

            // Create my Kernel
            auto scaleKernel = op::OpenCL::getInstance(0)->getKernelFunctorFromManager
                    <ScaleFunctor, float>(
                        "scaleKernel",scaleKernelString);

            // Write image to GPU from Caffe
            auto* gpuImagePtr = upCaffeNet->blobs().at(0)->mutable_gpu_data();
            cl::Buffer imageBuffer = cl::Buffer((cl_mem)gpuImagePtr, true);
            op::OpenCL::getInstance(0)->getQueue().enqueueWriteBuffer(imageBuffer, true, 0,
                                                                      imgResize.size().width * imgResize.size().height * sizeof(float),
                                                                      &imgFloat.at<float>(0));

            // Run a Kernel (Scale down intensity by 0.5)
            scaleKernel(cl::EnqueueArgs(op::OpenCL::getInstance(0)->getQueue(),
                                             cl::NDRange(imgResize.size().width, imgResize.size().height, 1)),
                                             imageBuffer, imgResize.size().width, imgResize.size().height, 0.5);

            // Read back image to GPU
            finalImage = cv::Mat(imgResize.size(),CV_32FC1);
            op::OpenCL::getInstance(0)->getQueue().enqueueReadBuffer(imageBuffer, CL_TRUE, 0,
                                                                     imgResize.size().width * imgResize.size().height * sizeof(float), &finalImage.at<float>(0));

            // Read subbuffer
            cl_buffer_region sourceRegion;
            op::OpenCL::getBufferRegion<float>(sourceRegion, 0, imgResize.size().width * imgResize.size().height / 2);
            cl::Buffer sourceBuffer = imageBuffer.createSubBuffer(CL_MEM_READ_ONLY,
                                                                  CL_BUFFER_CREATE_TYPE_REGION,
                                                                  &sourceRegion);


        }
        #if defined(USE_OPENCL) && defined(CL_HPP_ENABLE_EXCEPTIONS)
        catch (const cl::Error& e)
        {
            op::error(std::string(e.what()) + " : " + op::OpenCL::clErrorToString(e.err()) + " ID: " +
                  std::to_string(0), __LINE__, __FUNCTION__, __FILE__);
        }
        #endif
        catch (const std::exception& e)
        {
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }

        cv::imshow("win", finalImage);
        cv::waitKey(0);

        // Load model

        //            img.convertTo(img, CV_32FC1);
        //            img = cpuResize(img, cv::Size(img.size().width/4,img.size().height/4));
        //            img*=0.005;

        //            cv::Mat gpuImg = gpuResize(img, cv::Size(img.size().width*8,img.size().height*8));
        //            cv::Mat cpuImg = cpuResize(img, cv::Size(img.size().width*8,img.size().height*8));
        //            cv::imshow("gpuImg", gpuImg);
        //            cv::imshow("cpuImg", cpuImg);

        //            op::log("Done");
        //            cv::waitKey(0);

        return 0;
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }
}
#endif

int main(int argc, char *argv[])
{
#ifdef USE_OPENCL
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running handFromJsonTest
    return clTest();
#else
    op::error("OpenPose must be compiled with the `USE_CAFFE` & `USE_OPENCL` macro definitions in order to run"
              " this functionality.", __LINE__, __FUNCTION__, __FILE__);
    return 0;
#endif
}
