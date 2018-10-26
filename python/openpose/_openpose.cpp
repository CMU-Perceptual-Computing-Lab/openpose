#ifndef OPENPOSE_PYTHON_HPP
#define OPENPOSE_PYTHON_HPP
#define BOOST_DATE_TIME_NO_LIB

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <caffe/caffe.hpp>
#include <stdlib.h>

//#include <openpose/net/bodyPartConnectorCaffe.hpp>
//#include <openpose/net/nmsCaffe.hpp>
//#include <openpose/net/resizeAndMergeCaffe.hpp>
//#include <openpose/pose/poseParameters.hpp>
//#include <openpose/pose/enumClasses.hpp>
//#include <openpose/pose/poseExtractor.hpp>
//#include <openpose/gpu/cuda.hpp>
//#include <openpose/gpu/opencl.hcl>
//#include <openpose/core/macros.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <stdexcept>

#ifdef _WIN32
    #define OP_EXPORT __declspec(dllexport)
#else
    #define OP_EXPORT
#endif

namespace py = pybind11;

//typdef typename op::Array<float> op::Array_float;

int add(int i, int j) {
    return i + j;

    op::Array<float> xx;

    op::Datum d;
}

std::shared_ptr<op::Datum> getDatum(){

    std::shared_ptr<op::Datum> datum2 = std::make_shared<op::Datum>();

    datum2->outputData = op::Array<float>({2,2},1);

    return datum2;
}

void checkDatum(op::Datum* datum){
    std::cout << datum->outputData << std::endl;

    std::cout << datum->outputData.spData.use_count() << std::endl;

//    std::cout << datum->cvInputData.size() << std::endl;

//    cv::imshow("win",datum->cvInputData);

//    cv::waitKey(0);

//    cv::Mat x(rows, cols, CV_8UC3);

    //cv::Mat_<cv::Vec3b> l;
}


//op::Datum getDatum(){
//    op::Datum datum;
//    return datum;
//}

// PROBLEM. CV MAT ALLOWS DIRECT CASTING, BUT OP ARRAY DOES NOT!! NO METHOD TO PASS PTR

PYBIND11_MODULE(_openpose, m) {
    m.def("add", &add, "A function which adds two numbers",
          py::arg("i") = 1, py::arg("j") = 2);

    m.def("getDatum", &getDatum, "");
    m.def("checkDatum", &checkDatum, "");

    py::class_<op::Datum, std::shared_ptr<op::Datum>>(m, "Datum")
        .def(py::init<>())
        .def_readwrite("outputData", &op::Datum::outputData)
        .def_readwrite("cvInputData", &op::Datum::cvInputData)
        //.def("setName", &Pet::setName)
        //.def("getName", &Pet::getName)
            ;

//    py::class_<cv::Mat>(m, "Mat")
//        .def(py::init<>())
//        .def(py::init<const int&, const int&, const int&>(),
//             py::arg("rows"), py::arg("cols"), py::arg("type")=CV_8UC3)
//        //.def("setName", &Pet::setName)
//        //.def("getName", &Pet::getName)
//            ;

//    py::class_<op::Array<float>>(m, "Array", py::buffer_protocol())
//       .def("__repr__", [](op::Array<float> &a) { return a.toString(); })
//       .def("getSize", [](op::Array<float> &a) { return a.getSize(); })
//       .def(py::init<const std::vector<int>&>())
//       .def_buffer([](op::Array<float> &m) -> py::buffer_info {
//            return py::buffer_info(
//                m.getPtr(),                             /* Pointer to buffer */
//                sizeof(float),                          /* Size of one scalar */
//                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
//                m.getSize().size(),                     /* Number of dimensions */
//                m.getSize(),                            /* Buffer dimensions */
//                m.getStride()                          /* Strides (in bytes) for each index */
//            );
//        })
//    ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

namespace pybind11 { namespace detail {

template <> struct type_caster<op::Array<float>> {
    public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(op::Array<float>, _("numpy.ndarray"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool imp)
        {
            std::cout << "opArraay load" << std::endl;

            array b(src, true);
            buffer_info info = b.request();

            //std::vector<int> a(info.shape);
            std::vector<int> shape(std::begin(info.shape), std::end(info.shape));

            value = op::Array<float>(shape,0);

            value.spData.reset((float*)info.ptr);

            //memcpy(value.spData.get(), info.ptr, value.getVolume()*sizeof(float));

            return true;

            //exit(-1);

//            /* Try a default converting into a Python */
//            array b(src, true);
//            buffer_info info = b.request();

//            int ndims = info.ndim;

//            decltype(CV_32F) dtype;
//            size_t elemsize;
//            if (info.format == format_descriptor<float>::format()) {
//                if (ndims == 3) {
//                    dtype = CV_32FC3;
//                } else {
//                    dtype = CV_32FC1;
//                }
//                elemsize = sizeof(float);
//            } else if (info.format == format_descriptor<double>::format()) {
//                if (ndims == 3) {
//                    dtype = CV_64FC3;
//                } else {
//                    dtype = CV_64FC1;
//                }
//                elemsize = sizeof(double);
//            } else if (info.format == format_descriptor<unsigned char>::format()) {
//                if (ndims == 3) {
//                    dtype = CV_8UC3;
//                } else {
//                    dtype = CV_8UC1;
//                }
//                elemsize = sizeof(unsigned char);
//            } else {
//                throw std::logic_error("Unsupported type");
//                return false;
//            }

//            std::vector<int> shape = {info.shape[0], info.shape[1]};

//            value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
//            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an inty instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(const op::Array<float> &m, return_value_policy, handle defval)
        {
            std::string format = format_descriptor<float>::format();
            return array(buffer_info(
                m.spData.get(),     /* Pointer to buffer */
                sizeof(float),       /* Size of one scalar */
                format,         /* Python struct-style format descriptor */
                m.getSize().size(),            /* Number of dimensions */
                m.getSize(),      /* Buffer dimensions */
                m.getStride()         /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail


namespace pybind11 { namespace detail {

template <> struct type_caster<cv::Mat> {
    public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool)
        {
            /* Try a default converting into a Python */
            array b(src, true);
            buffer_info info = b.request();

            int ndims = info.ndim;

            decltype(CV_32F) dtype;
            size_t elemsize;
            if (info.format == format_descriptor<float>::format()) {
                if (ndims == 3) {
                    dtype = CV_32FC3;
                } else {
                    dtype = CV_32FC1;
                }
                elemsize = sizeof(float);
            } else if (info.format == format_descriptor<double>::format()) {
                if (ndims == 3) {
                    dtype = CV_64FC3;
                } else {
                    dtype = CV_64FC1;
                }
                elemsize = sizeof(double);
            } else if (info.format == format_descriptor<unsigned char>::format()) {
                if (ndims == 3) {
                    dtype = CV_8UC3;
                } else {
                    dtype = CV_8UC1;
                }
                elemsize = sizeof(unsigned char);
            } else {
                throw std::logic_error("Unsupported type");
                return false;
            }

            std::vector<int> shape = {info.shape[0], info.shape[1]};

            value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an inty instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(const cv::Mat &m, return_value_policy, handle defval)
        {
            std::cout << "m.cols : " << m.cols << std::endl;
            std::cout << "m.rows : " << m.rows << std::endl;
            std::string format = format_descriptor<unsigned char>::format();
            size_t elemsize = sizeof(unsigned char);
            int dim;
            switch(m.type()) {
                case CV_8U:
                    format = format_descriptor<unsigned char>::format();
                    elemsize = sizeof(unsigned char);
                    dim = 2;
                    break;
                case CV_8UC3:
                    format = format_descriptor<unsigned char>::format();
                    elemsize = sizeof(unsigned char);
                    dim = 3;
                    break;
                case CV_32F:
                    format = format_descriptor<float>::format();
                    elemsize = sizeof(float);
                    dim = 2;
                    break;
                case CV_64F:
                    format = format_descriptor<double>::format();
                    elemsize = sizeof(double);
                    dim = 2;
                    break;
                default:
                    throw std::logic_error("Unsupported type");
            }

            std::vector<size_t> bufferdim;
            std::vector<size_t> strides;
            if (dim == 2) {
                bufferdim = {(size_t) m.rows, (size_t) m.cols};
                strides = {elemsize * (size_t) m.cols, elemsize};
            } else if (dim == 3) {
                bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) 3};
                strides = {(size_t) elemsize * m.cols * 3, (size_t) elemsize * 3, (size_t) elemsize};
            }
            return array(buffer_info(
                m.data,         /* Pointer to buffer */
                elemsize,       /* Size of one scalar */
                format,         /* Python struct-style format descriptor */
                dim,            /* Number of dimensions */
                bufferdim,      /* Buffer dimensions */
                strides         /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail


//#define default_logging_level 3
//#define default_output_resolution "-1x-1"
//#define default_net_resolution "-1x368"
//#define default_model_pose "COCO"
//#define default_alpha_pose 0.6
//#define default_scale_gap 0.3
//#define default_scale_number 1
//#define default_render_threshold 0.05
//#define default_num_gpu_start 0
//#define default_disable_blending false
//#define default_model_folder "models/"

//// Todo, have GPU Number, handle, OpenCL/CPU Cases
//OP_API class OpenPose {
//public:
//    std::unique_ptr<op::PoseExtractorCaffe> poseExtractorCaffe;
//    std::unique_ptr<op::PoseCpuRenderer> poseRenderer;
//    std::unique_ptr<op::FrameDisplayer> frameDisplayer;
//    std::unique_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor;

//    std::unique_ptr<op::ResizeAndMergeCaffe<float>> resizeAndMergeCaffe;
//    std::unique_ptr<op::NmsCaffe<float>> nmsCaffe;
//    std::unique_ptr<op::BodyPartConnectorCaffe<float>> bodyPartConnectorCaffe;
//    std::shared_ptr<caffe::Blob<float>> heatMapsBlob;
//    std::shared_ptr<caffe::Blob<float>> peaksBlob;
//    op::Array<float> mPoseKeypoints;
//    op::Array<float> mPoseScores;
//    op::PoseModel poseModel;
//    int mGpuID;

//    OpenPose(int FLAGS_logging_level = default_logging_level,
//        std::string FLAGS_output_resolution = default_output_resolution,
//        std::string FLAGS_net_resolution = default_net_resolution,
//        std::string FLAGS_model_pose = default_model_pose,
//        float FLAGS_alpha_pose = default_alpha_pose,
//        float FLAGS_scale_gap = default_scale_gap,
//        int FLAGS_scale_number = default_scale_number,
//        float FLAGS_render_threshold = default_render_threshold,
//        int FLAGS_num_gpu_start = default_num_gpu_start,
//        int FLAGS_disable_blending = default_disable_blending,
//        std::string FLAGS_model_folder = default_model_folder
//    ) {
//        mGpuID = FLAGS_num_gpu_start;
//#ifdef USE_CUDA
//        caffe::Caffe::set_mode(caffe::Caffe::GPU);
//        caffe::Caffe::SetDevice(mGpuID);
//#elif USE_OPENCL
//        caffe::Caffe::set_mode(caffe::Caffe::GPU);
//        std::vector<int> devices;
//        const int maxNumberGpu = op::OpenCL::getTotalGPU();
//        for (auto i = 0; i < maxNumberGpu; i++)
//            devices.emplace_back(i);
//        caffe::Caffe::SetDevices(devices);
//        caffe::Caffe::SelectDevice(mGpuID, true);
//        op::OpenCL::getInstance(mGpuID, CL_DEVICE_TYPE_GPU, true);
//#else
//        caffe::Caffe::set_mode(caffe::Caffe::CPU);
//#endif
//        op::log("OpenPose Library Python Wrapper", op::Priority::High);
//        // ------------------------- INITIALIZATION -------------------------
//        // Step 1 - Set logging level
//        // - 0 will output all the logging messages
//        // - 255 will output nothing
//        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
//        op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
//        // Step 2 - Read GFlags (user defined configuration)
//        // outputSize
//        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
//        // netInputSize
//        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
//        // poseModel
//        poseModel = op::flagsToPoseModel(FLAGS_model_pose);
//        // Check no contradictory flags enabled
//        if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
//            op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
//        if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
//            op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
//                __LINE__, __FUNCTION__, __FILE__);
//        // Step 3 - Initialize all required classes
//        scaleAndSizeExtractor = std::unique_ptr<op::ScaleAndSizeExtractor>(new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap));

//        poseExtractorCaffe = std::unique_ptr<op::PoseExtractorCaffe>(new op::PoseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start });

//        poseRenderer = std::unique_ptr<op::PoseCpuRenderer>(new op::PoseCpuRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
//            (float)FLAGS_alpha_pose });
//        frameDisplayer = std::unique_ptr<op::FrameDisplayer>(new op::FrameDisplayer{ "OpenPose Tutorial - Example 1", outputSize });

//        // Custom
//        resizeAndMergeCaffe = std::unique_ptr<op::ResizeAndMergeCaffe<float>>(new op::ResizeAndMergeCaffe<float>{});
//        nmsCaffe = std::unique_ptr<op::NmsCaffe<float>>(new op::NmsCaffe<float>{});
//        bodyPartConnectorCaffe = std::unique_ptr<op::BodyPartConnectorCaffe<float>>(new op::BodyPartConnectorCaffe<float>{});
//        heatMapsBlob = { std::make_shared<caffe::Blob<float>>(1,1,1,1) };
//        peaksBlob = { std::make_shared<caffe::Blob<float>>(1,1,1,1) };
//        bodyPartConnectorCaffe->setPoseModel(poseModel);

//        // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
//        poseExtractorCaffe->initializationOnThread();
//        poseRenderer->initializationOnThread();
//    }

//    std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
//        std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
//    {
//        try
//        {
//            // Prepare spCaffeNetOutputBlobss
//            std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
//            for (auto i = 0u; i < caffeNetOutputBlobs.size(); i++)
//                caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
//            return caffeNetOutputBlobs;
//        }
//        catch (const std::exception& e)
//        {
//            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
//            return{};
//        }
//    }

//    void forward(const cv::Mat& inputImage, op::Array<float>& poseKeypoints, cv::Mat& displayImage, bool display = false) {
//        op::OpOutputToCvMat opOutputToCvMat;
//        op::CvMatToOpInput cvMatToOpInput;
//        op::CvMatToOpOutput cvMatToOpOutput;
//        if (inputImage.empty())
//            op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
//        const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
//        // Step 2 - Get desired scale sizes
//        std::vector<double> scaleInputToNetInputs;
//        std::vector<op::Point<int>> netInputSizes;
//        double scaleInputToOutput;
//        op::Point<int> outputResolution;
//        std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
//            = scaleAndSizeExtractor->extract(imageSize);
//        // Step 3 - Format input image to OpenPose input and output formats
//        const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);

//        // Step 4 - Estimate poseKeypoints
//        poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
//        poseKeypoints = poseExtractorCaffe->getPoseKeypoints();

//        if (display) {
//            auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
//            // Step 5 - Render poseKeypoints
//            poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
//            // Step 6 - OpenPose output format to cv::Mat
//            displayImage = opOutputToCvMat.formatToCvMat(outputArray);
//        }
//    }

//    void poseFromHeatmap(const cv::Mat& inputImage, std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob, op::Array<float>& poseKeypoints, cv::Mat& displayImage, std::vector<op::Point<int>>& imageSizes) {
//        // Get Scale
//        const op::Point<int> inputDataSize{ inputImage.cols, inputImage.rows };

//        // Convert to Ptr
//        //std::vector<boost::shared_ptr<caffe::Blob<float>>> a;
//        //caffeNetOutputBlob.emplace_back(caffeHmPtr);
//        const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);

//        // To be called once only
//        resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, { heatMapsBlob.get() },
//            op::getPoseNetDecreaseFactor(poseModel), 1.f / 1.f, true,
//            0);
//        nmsCaffe->Reshape({ heatMapsBlob.get() }, { peaksBlob.get() }, op::getPoseMaxPeaks(poseModel),
//            op::getPoseNumberBodyParts(poseModel), 0);
//        bodyPartConnectorCaffe->Reshape({ heatMapsBlob.get(), peaksBlob.get() });

//        // Normal
//        op::OpOutputToCvMat opOutputToCvMat;
//        op::CvMatToOpInput cvMatToOpInput;
//        op::CvMatToOpOutput cvMatToOpOutput;
//        if (inputImage.empty())
//            op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
//        const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };
//        // Step 2 - Get desired scale sizes
//        std::vector<double> scaleInputToNetInputs;
//        std::vector<op::Point<int>> netInputSizes;
//        double scaleInputToOutput;
//        op::Point<int> outputResolution;

//        std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
//            = scaleAndSizeExtractor->extract(imageSize);

//        const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);

//        // Run the modes
//        const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
//        resizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
//        std::vector<caffe::Blob<float>*> heatMapsBlobs{ heatMapsBlob.get() };
//        std::vector<caffe::Blob<float>*> peaksBlobs{ peaksBlob.get() };
//#ifdef USE_CUDA
//        resizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
//#elif USE_OPENCL
//        resizeAndMergeCaffe->Forward_ocl(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
//#else
//        resizeAndMergeCaffe->Forward_cpu(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
//#endif

//        nmsCaffe->setThreshold((float)poseExtractorCaffe->get(op::PoseProperty::NMSThreshold));
//#ifdef USE_CUDA
//        nmsCaffe->Forward_gpu(heatMapsBlobs, peaksBlobs);// ~2ms
//#elif USE_OPENCL
//        nmsCaffe->Forward_ocl(heatMapsBlobs, peaksBlobs);// ~2ms
//#else
//        nmsCaffe->Forward_cpu(heatMapsBlobs, peaksBlobs);// ~2ms
//#endif
//        op::cudaCheck(__LINE__, __FUNCTION__, __FILE__);

//        float mScaleNetToOutput = 1. / scaleInputToNetInputs[0];
//        bodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
//        bodyPartConnectorCaffe->setInterMinAboveThreshold(
//            (float)poseExtractorCaffe->get(op::PoseProperty::ConnectInterMinAboveThreshold)
//        );
//        bodyPartConnectorCaffe->setInterThreshold((float)poseExtractorCaffe->get(op::PoseProperty::ConnectInterThreshold));
//        bodyPartConnectorCaffe->setMinSubsetCnt((int)poseExtractorCaffe->get(op::PoseProperty::ConnectMinSubsetCnt));
//        bodyPartConnectorCaffe->setMinSubsetScore((float)poseExtractorCaffe->get(op::PoseProperty::ConnectMinSubsetScore));

//#ifdef USE_CUDA
//        bodyPartConnectorCaffe->Forward_gpu({ heatMapsBlob.get(),
//            peaksBlob.get() },
//            mPoseKeypoints, mPoseScores);
//#else
//        bodyPartConnectorCaffe->Forward_cpu({ heatMapsBlob.get(),
//            peaksBlob.get() },
//            mPoseKeypoints, mPoseScores);
//#endif
//        poseKeypoints = mPoseKeypoints;

//        auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
//        // Step 5 - Render poseKeypoints
//        poseRenderer->renderPose(outputArray, mPoseKeypoints, scaleInputToOutput);
//        // Step 6 - OpenPose output format to cv::Mat
//        displayImage = opOutputToCvMat.formatToCvMat(outputArray);
//    }
//};

//#ifdef __cplusplus
//extern "C" {
//#endif

//    typedef void* c_OP;
//    op::Array<float> output;

//    OP_EXPORT c_OP newOP(int logging_level,
//        char* output_resolution,
//        char* net_resolution,
//        char* model_pose,
//        float alpha_pose,
//        float scale_gap,
//        int scale_number,
//        float render_threshold,
//        int num_gpu_start,
//        bool disable_blending,
//        char* model_folder
//    ) {
//        return new OpenPose(logging_level, output_resolution, net_resolution, model_pose, alpha_pose,
//            scale_gap, scale_number, render_threshold, num_gpu_start, disable_blending, model_folder);
//    }
//    OP_EXPORT void delOP(c_OP op) {
//        delete (OpenPose *)op;
//    }
//    OP_EXPORT void forward(c_OP op, unsigned char* img, size_t rows, size_t cols, int* size, unsigned char* displayImg, bool display) {
//        OpenPose* openPose = (OpenPose*)op;
//        cv::Mat image(rows, cols, CV_8UC3, img);
//        cv::Mat displayImage(rows, cols, CV_8UC3, displayImg);
//        openPose->forward(image, output, displayImage, display);
//        if (output.getSize().size()) {
//            size[0] = output.getSize()[0];
//            size[1] = output.getSize()[1];
//            size[2] = output.getSize()[2];
//        }
//        else {
//            size[0] = 0; size[1] = 0; size[2] = 0;
//        }
//        if (display) memcpy(displayImg, displayImage.ptr(), sizeof(unsigned char)*rows*cols * 3);
//    }
//    OP_EXPORT void getOutputs(c_OP op, float* array) {
//        if (output.getSize().size())
//            memcpy(array, output.getPtr(), output.getSize()[0] * output.getSize()[1] * output.getSize()[2] * sizeof(float));
//    }
//    OP_EXPORT void poseFromHeatmap(c_OP op, unsigned char* img, size_t rows, size_t cols, unsigned char* displayImg, float* hm, int* size, float* ratios) {
//        OpenPose* openPose = (OpenPose*)op;
//        cv::Mat image(rows, cols, CV_8UC3, img);
//        cv::Mat displayImage(rows, cols, CV_8UC3, displayImg);

//        std::vector<boost::shared_ptr<caffe::Blob<float>>> caffeNetOutputBlob;

//        for (int i = 0; i<size[0]; i++) {
//            boost::shared_ptr<caffe::Blob<float>> caffeHmPtr(new caffe::Blob<float>());
//            caffeHmPtr->Reshape(1, size[1], size[2] * ((float)ratios[i] / (float)ratios[0]), size[3] * ((float)ratios[i] / (float)ratios[0]));
//            float* startIndex = &hm[i*size[1] * size[2] * size[3]];
//            for (int d = 0; d<caffeHmPtr->shape()[1]; d++) {
//                for (int r = 0; r<caffeHmPtr->shape()[2]; r++) {
//                    for (int c = 0; c<caffeHmPtr->shape()[3]; c++) {
//                        int toI = d*caffeHmPtr->shape()[2] * caffeHmPtr->shape()[3] + r*caffeHmPtr->shape()[3] + c;
//                        int fromI = d*size[2] * size[3] + r*size[3] + c;
//                        caffeHmPtr->mutable_cpu_data()[toI] = startIndex[fromI];
//                    }
//                }
//            }
//            caffeNetOutputBlob.emplace_back(caffeHmPtr);
//        }

//        std::vector<op::Point<int>> imageSizes;
//        for (int i = 0; i<size[0]; i++) {
//            op::Point<int> point(cols*ratios[i], rows*ratios[i]);
//            imageSizes.emplace_back(point);
//        }

//        openPose->poseFromHeatmap(image, caffeNetOutputBlob, output, displayImage, imageSizes);
//        memcpy(displayImg, displayImage.ptr(), sizeof(unsigned char)*rows*cols * 3);
//        // Copy back kp size
//        if (output.getSize().size()) {
//            size[0] = output.getSize()[0];
//            size[1] = output.getSize()[1];
//            size[2] = output.getSize()[2];
//        }
//        else {
//            size[0] = 0; size[1] = 0; size[2] = 0;
//        }
//    }

//#ifdef __cplusplus
//}
//#endif

#endif
