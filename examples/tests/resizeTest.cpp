// ------------------------- OpenPose Resize Layer Testing -------------------------

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

#ifdef USE_CUDA
    #ifdef USE_CAFFE
        #include <caffe/net.hpp>
    #endif

    DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");

    cv::Mat gpuResize(cv::Mat& img, const cv::Size& newSize)
    {
        #if defined USE_CAFFE && defined USE_CUDA
            // Upload to Source to GPU
            float* cpuPtr = &img.at<float>(0);
            float* gpuPtr;
            cudaMallocHost((void **)&gpuPtr, img.size().width * img.size().height * sizeof(float));
            cudaMemcpy(gpuPtr, cpuPtr, img.size().width * img.size().height * sizeof(float),
                       cudaMemcpyHostToDevice);

            // Upload to Dest to GPU
            cv::Mat newImg = cv::Mat(newSize,CV_32FC1,cv::Scalar(0));
            float* newCpuPtr = &newImg.at<float>(0);
            float* newGpuPtr;
            cudaMallocHost((void **)&newGpuPtr, newSize.width * newSize.height * sizeof(float));
            cudaMemcpy(newGpuPtr, newCpuPtr, newSize.width * newSize.height * sizeof(float),
                       cudaMemcpyHostToDevice);

            std::vector<const float*> sourcePtrs;
            sourcePtrs.emplace_back(gpuPtr);
            std::array<int, 4> targetSize = {1,1,newImg.size().height,newImg.size().width};
            std::array<int, 4> sourceSize = {1,1,img.size().height,img.size().width};
            std::vector<std::array<int, 4>> sourceSizes;
            sourceSizes.emplace_back(sourceSize);
            op::resizeAndMergeGpu(newGpuPtr, sourcePtrs, targetSize, sourceSizes);
            cudaMemcpy(newCpuPtr, newGpuPtr, newImg.size().width * newImg.size().height * sizeof(float),
                       cudaMemcpyDeviceToHost);

            cudaFree(gpuPtr);
            cudaFree(newGpuPtr);
            return newImg;
        #else
            UNUSED(img);
            UNUSED(newSize);
            op::error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
                  " this functionality.", __LINE__, __FUNCTION__, __FILE__);
            return cv::Mat();
        #endif
    }

    cv::Mat cpuResize(cv::Mat& img, cv::Size newSize)
    {
        // Upload to Source to GPU
        float* cpuPtr = &img.at<float>(0);

        // Upload to Dest to GPU
        cv::Mat newImg = cv::Mat(newSize,CV_32FC1,cv::Scalar(0));

        std::vector<const float*> sourcePtrs;
        sourcePtrs.emplace_back(cpuPtr);
        std::array<int, 4> targetSize = {1,1,newImg.size().height,newImg.size().width};
        std::array<int, 4> sourceSize = {1,1,img.size().height,img.size().width};
        std::vector<std::array<int, 4>> sourceSizes;
        sourceSizes.emplace_back(sourceSize);
        op::resizeAndMergeCpu(&newImg.at<float>(0), sourcePtrs, targetSize, sourceSizes);

        return newImg;
    }

    int resizeTest()
    {
        try
        {
            // logging_level
            op::Matrix opImg = op::loadImage(FLAGS_image_path, op::getCvLoadImageGrayScale());
            cv::Mat img = OP_OP2CVMAT(opImg);
            if(img.empty())
                op::error("Could not open or find the image: " + FLAGS_image_path, __LINE__, __FUNCTION__, __FILE__);
            img.convertTo(img, CV_32FC1);
            img = cpuResize(img, cv::Size(img.size().width/4,img.size().height/4));
            img*=0.005;

            cv::Mat gpuImg = gpuResize(img, cv::Size(img.size().width*8,img.size().height*8));
            cv::Mat cpuImg = cpuResize(img, cv::Size(img.size().width*8,img.size().height*8));
            cv::imshow("gpuImg", gpuImg);
            cv::imshow("cpuImg", cpuImg);

            op::opLog("Done");
            cv::waitKey(0);

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
    #ifdef USE_CUDA
        // Parsing command line flags
        gflags::ParseCommandLineFlags(&argc, &argv, true);

        // Running handFromJsonTest
        return resizeTest();
    #else
        op::error("OpenPose must be compiled with the `USE_CAFFE` & `USE_CUDA` macro definitions in order to run"
              " this functionality.", __LINE__, __FUNCTION__, __FILE__);
        return 0;
    #endif
}
