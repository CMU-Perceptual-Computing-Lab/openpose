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

template<typename Dtype>
void matToCaffe(Dtype* caffeImg, const cv::Mat& imgAug){
    const int imageAugmentedArea = imgAug.rows * imgAug.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imgAug.data);
    //caffeImg = new Dtype[imgAug.channels()*imgAug.size().width*imgAug.size().height];
    for (auto y = 0; y < imgAug.rows; y++)
    {
        const auto yOffset = y*imgAug.cols;
        for (auto x = 0; x < imgAug.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
            auto* bgr = &uCharPtrCvMat[3*xyOffset];
            caffeImg[xyOffset] = (bgr[0] - 128) / 256.0;
            caffeImg[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
            caffeImg[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
        }
    }
}

template<typename Dtype>
void caffeToMat(cv::Mat& img, const Dtype* caffeImg, cv::Size imageSize){
    // Need a function to convert back
    img = cv::Mat(imageSize, CV_8UC3);
    const int imageAugmentedArea = img.rows * img.cols;
    auto* imgPtr = (unsigned char*)(img.data);
    for (auto y = 0; y < img.rows; y++)
    {
        const auto yOffset = y*img.cols;
        for (auto x = 0; x < img.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            auto* bgr = &imgPtr[3*xyOffset];
            bgr[0] = (caffeImg[xyOffset]*256.) + 128;
            bgr[1] = (caffeImg[xyOffset + imageAugmentedArea]*256.) + 128;
            bgr[2] = (caffeImg[xyOffset + 2*imageAugmentedArea]*256.) + 128;
        }
    }
}

int clTest()
{
    try
    {
        // logging_level
        cv::Mat img = cv::imread(FLAGS_image_path);
        if(img.empty())
            op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
        cv::Mat imgResize; cv::resize(img, imgResize, cv::Size(368,368));
        cv::Mat imgFloat; imgResize.convertTo(imgFloat, CV_32FC3);
        imgFloat /= 255.;
        int imageVolume = imgFloat.size().width * imgFloat.size().height * imgFloat.channels();
        std::cout << imgFloat.channels() << std::endl;

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
        upCaffeNet->blobs()[0]->Reshape({1,imgFloat.channels(),imgResize.size().width,imgResize.size().height});
        upCaffeNet->Reshape();

        // GPU Test
        cv::Mat finalImage = imgFloat;
        try{

            // Create my Kernel
            auto scaleKernel = op::OpenCL::getInstance(0)->getKernelFunctorFromManager
                    <ScaleFunctor, float>(
                        "scaleKernel",scaleKernelString);

            // Write image to GPU from Caffe
            auto* gpuImagePtr = upCaffeNet->blobs().at(0)->mutable_gpu_data();
            cl::Buffer imageBuffer = cl::Buffer((cl_mem)gpuImagePtr, true);
            op::OpenCL::getInstance(0)->getQueue().enqueueWriteBuffer(imageBuffer, true, 0,
                                                                      imgResize.size().width * imgResize.size().height * imgFloat.channels() * sizeof(float),
                                                                      &imgFloat.at<float>(0));

            for(int i=0; i<imgFloat.channels() - 1; i++){

                // Read subbuffer
                cl_buffer_region sourceRegion;
                op::OpenCL::getBufferRegion<float>(sourceRegion, i * imgResize.size().width * imgResize.size().height, imgResize.size().width * imgResize.size().height);
                cl::Buffer regionBuffer = imageBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                      CL_BUFFER_CREATE_TYPE_REGION,
                                                                      &sourceRegion);

                // Run a Kernel (Scale down intensity by 0.5)
                scaleKernel(cl::EnqueueArgs(op::OpenCL::getInstance(0)->getQueue(),
                                                 cl::NDRange(imgResize.size().width, imgResize.size().height, 1)),
                                                 regionBuffer, imgResize.size().width, imgResize.size().height, 0.5);


            }

            // Read back image to GPU
            finalImage = cv::Mat(imgResize.size(),CV_32FC3);
            op::OpenCL::getInstance(0)->getQueue().enqueueReadBuffer(imageBuffer, CL_TRUE, 0,
                                                                     imgResize.size().width * imgResize.size().height * imgFloat.channels() * sizeof(float), &finalImage.at<float>(0));

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
    std::thread t(&clTest);
    t.join();
    return 0;
#else
    op::error("OpenPose must be compiled with the `USE_CAFFE` & `USE_OPENCL` macro definitions in order to run"
              " this functionality.", __LINE__, __FUNCTION__, __FILE__);
    return 0;
#endif
}
