// ------------------------- OpenPose Resize Layer Testing -------------------------

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
// OpenCL dependencies
#ifdef USE_OPENCL
#include <openpose/gpu/opencl.hcl>
#include <openpose/gpu/cl2.hpp>

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

        // Convert to caffe image
        caffe::BlobProto blob_proto;
        blob_proto.set_channels(3);
        blob_proto.set_height(imgResize.size().width);
        blob_proto.set_width(imgResize.size().height);
        blob_proto.clear_data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < imgResize.size().height; ++h) {
                for (int w = 0; w < imgResize.size().width; ++w) {
                    blob_proto.add_data(imgResize.at<cv::Vec3f>(h, w)[c]);
                }
            }
        }
        blob_proto.set_num(1);
        caffe::Blob<float>* input_layer = upCaffeNet->input_blobs()[0];
        input_layer->FromProto(blob_proto);
        upCaffeNet->Forward(0);

        boost::shared_ptr<caffe::Blob<float> > output_blob = upCaffeNet->blob_by_name("net_output");

        // Test
        cl::Device& device = op::OpenCL::getInstance(0)->getDevice();
        cl_uint mem_align;
        clGetDeviceInfo(device.get(), CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(mem_align), &mem_align, nullptr);
        std::cout << "Alignment in bits of the base address : " << mem_align << std::endl;

        // GPU Test
        cv::Mat finalImage = imgFloat;
        try{

            // Get
            float* gpuPtr = output_blob->mutable_gpu_data();
            cl::Buffer outputBuffer((cl_mem)gpuPtr, true);

            // Read it
            // Read back image to GPU
            float* heatmaps = new float[output_blob->shape()[1] * output_blob->shape()[2] * output_blob->shape()[3]];
            op::OpenCL::getInstance(0)->getQueue().enqueueReadBuffer(outputBuffer, CL_TRUE, 0,
                                                                     output_blob->shape()[1] * output_blob->shape()[2] * output_blob->shape()[3] * sizeof(float), heatmaps);

            int heatmapChannels = output_blob->shape()[1];
            int shape = output_blob->shape()[2] * output_blob->shape()[3];
            for(int i=0; i<heatmapChannels; i++){
                cv::Mat hm(cv::Size(output_blob->shape()[2], output_blob->shape()[3]), CV_32FC1);
                // Read subbuffer
                cl_buffer_region sourceRegion;
                op::OpenCL::getBufferRegion<float>(sourceRegion, i * shape, shape);
                cl::Buffer regionBuffer = outputBuffer.createSubBuffer(CL_MEM_READ_WRITE,
                                                                      CL_BUFFER_CREATE_TYPE_REGION,
                                                                      &sourceRegion);
            }
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

int main()
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
