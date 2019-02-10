#ifndef OPENPOSE_PYTHON_HPP
#define OPENPOSE_PYTHON_HPP
#define BOOST_DATE_TIME_NO_LIB

#include <openpose/flags.hpp>
#include <openpose/headers.hpp>
#include <openpose/wrapper/headers.hpp>

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

namespace op{

namespace py = pybind11;

void parse_gflags(const std::vector<std::string>& argv)
{
    std::vector<char*> argv_vec;
    for(auto& arg : argv) argv_vec.emplace_back((char*)arg.c_str());
    char** cast = &argv_vec[0];
    int size = argv_vec.size();
    gflags::ParseCommandLineFlags(&size, &cast, true);
}

void init_int(py::dict d)
{
    std::vector<std::string> argv;
    argv.emplace_back("openpose.py");
    for (auto item : d){
        argv.emplace_back("--" + std::string(py::str(item.first)));
        argv.emplace_back(py::str(item.second));
    }
    parse_gflags(argv);
}

void init_argv(std::vector<std::string> argv)
{
    argv.insert(argv.begin(), "openpose.py");
    parse_gflags(argv);
}

class WrapperPython{
public:
    std::unique_ptr<op::Wrapper> opWrapper;

    WrapperPython(int mode = 0)
    {
        op::log("Starting OpenPose Python Wrapper...", op::Priority::High);

        // Construct opWrapper
        opWrapper = std::unique_ptr<op::Wrapper>(new op::Wrapper(static_cast<op::ThreadManagerMode>(mode)));
    }

    void configure(py::dict params = py::dict())
    {
        if(params.size()) init_int(params);

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper->configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper->configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper->configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper->configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_foot_json, FLAGS_write_coco_json_variant,
            FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
            FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
            FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapper->configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper->disableMultiThreading();
    }

    void start(){
        opWrapper->start();
    }

    void stop(){
        opWrapper->stop();
    }

    void exec(){
        const auto cameraSize = op::flagsToPoint(FLAGS_camera_resolution, "-1x-1");
        op::ProducerType producerType;
        std::string producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera, FLAGS_flir_camera, FLAGS_flir_camera_index);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, FLAGS_camera_parameter_path, FLAGS_frame_undistort, FLAGS_3d_views};
        opWrapper->configure(wrapperStructInput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapper->configure(wrapperStructGui);
        opWrapper->exec();
    }

    void emplaceAndPop(std::vector<std::shared_ptr<op::Datum>>& l)
    {
        auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>(l);
        opWrapper->emplaceAndPop(datumsPtr);
    }

    void waitAndEmplace(std::vector<std::shared_ptr<op::Datum>>& l)
    {
        auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>(l);
        opWrapper->waitAndEmplace(datumsPtr);
    }

    bool waitAndPop(std::vector<std::shared_ptr<op::Datum>>& l)
    {
        auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>(l);
        return opWrapper->waitAndPop(datumsPtr);
    }
};

std::vector<std::string> getImagesFromDirectory(const std::string& directoryPath)
{
    return op::getFilesOnDirectory(directoryPath, op::Extensions::Images);
}

PYBIND11_MODULE(pyopenpose, m) {

    // Functions for Init Params
    m.def("init_int", &init_int, "Init Function");
    m.def("init_argv", &init_argv, "Init Function");
    m.def("get_gpu_number", &op::getGpuNumber, "Get Total GPU");
    m.def("get_images_on_directory", &op::getImagesFromDirectory, "Get Images On Directory");

    // OpenposePython
    py::class_<WrapperPython>(m, "WrapperPython")
        .def(py::init<>())
        .def(py::init<int>())
        .def("configure", &WrapperPython::configure)
        .def("start", &WrapperPython::start)
        .def("stop", &WrapperPython::stop)
        .def("execute", &WrapperPython::exec)
        .def("emplaceAndPop", &WrapperPython::emplaceAndPop)
        .def("waitAndEmplace", &WrapperPython::waitAndEmplace)
        .def("waitAndPop", &WrapperPython::waitAndPop)
        ;

    // Datum Object
    py::class_<op::Datum, std::shared_ptr<op::Datum>>(m, "Datum")
        .def(py::init<>())
        .def_readwrite("id", &op::Datum::id)
        .def_readwrite("subId", &op::Datum::subId)
        .def_readwrite("subIdMax", &op::Datum::subIdMax)
        .def_readwrite("name", &op::Datum::name)
        .def_readwrite("frameNumber", &op::Datum::frameNumber)
        .def_readwrite("cvInputData", &op::Datum::cvInputData)
        .def_readwrite("inputNetData", &op::Datum::inputNetData)
        .def_readwrite("outputData", &op::Datum::outputData)
        .def_readwrite("cvOutputData", &op::Datum::cvOutputData)
        .def_readwrite("cvOutputData3D", &op::Datum::cvOutputData3D)
        .def_readwrite("poseKeypoints", &op::Datum::poseKeypoints)
        .def_readwrite("poseIds", &op::Datum::poseIds)
        .def_readwrite("poseScores", &op::Datum::poseScores)
        .def_readwrite("poseHeatMaps", &op::Datum::poseHeatMaps)
        .def_readwrite("poseCandidates", &op::Datum::poseCandidates)
        .def_readwrite("faceRectangles", &op::Datum::faceRectangles)
        .def_readwrite("faceKeypoints", &op::Datum::faceKeypoints)
        .def_readwrite("faceHeatMaps", &op::Datum::faceHeatMaps)
        .def_readwrite("handRectangles", &op::Datum::handRectangles)
        .def_readwrite("handKeypoints", &op::Datum::handKeypoints)
        .def_readwrite("handHeatMaps", &op::Datum::handHeatMaps)
        .def_readwrite("poseKeypoints3D", &op::Datum::poseKeypoints3D)
        .def_readwrite("faceKeypoints3D", &op::Datum::faceKeypoints3D)
        .def_readwrite("handKeypoints3D", &op::Datum::handKeypoints3D)
        .def_readwrite("cameraMatrix", &op::Datum::cameraMatrix)
        .def_readwrite("cameraExtrinsics", &op::Datum::cameraExtrinsics)
        .def_readwrite("cameraIntrinsics", &op::Datum::cameraIntrinsics)
        .def_readwrite("poseNetOutput", &op::Datum::poseNetOutput)
        .def_readwrite("scaleInputToNetInputs", &op::Datum::scaleInputToNetInputs)
        .def_readwrite("netInputSizes", &op::Datum::netInputSizes)
        .def_readwrite("scaleInputToOutput", &op::Datum::scaleInputToOutput)
        .def_readwrite("netOutputSize", &op::Datum::netOutputSize)
        .def_readwrite("scaleNetToOutput", &op::Datum::scaleNetToOutput)
        .def_readwrite("elementRendered", &op::Datum::elementRendered)
        ;

    // Rectangle
    py::class_<op::Rectangle<float>>(m, "Rectangle")
        .def("__repr__", [](op::Rectangle<float> &a) { return a.toString(); })
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &op::Rectangle<float>::x)
        .def_readwrite("y", &op::Rectangle<float>::y)
        .def_readwrite("width", &op::Rectangle<float>::width)
        .def_readwrite("height", &op::Rectangle<float>::height)
        ;

    // Point
    py::class_<op::Point<int>>(m, "Point")
        .def("__repr__", [](op::Point<int> &a) { return a.toString(); })
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("x", &op::Point<int>::x)
        .def_readwrite("y", &op::Point<int>::y)
        ;

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

}

// Numpy - op::Array<float> interop
namespace pybind11 { namespace detail {

template <> struct type_caster<op::Array<float>> {
    public:

        PYBIND11_TYPE_CASTER(op::Array<float>, _("numpy.ndarray"));

        // Cast numpy to op::Array<float>
        bool load(handle src, bool imp)
        {
            // array b(src, true);
            array b = reinterpret_borrow<array>(src);
            buffer_info info = b.request();

            if (info.format != format_descriptor<float>::format())
                throw std::runtime_error("op::Array only supports float32 now");

            //std::vector<int> a(info.shape);
            std::vector<int> shape(std::begin(info.shape), std::end(info.shape));

            // No copy
            value = op::Array<float>(shape, (float*)info.ptr);
            // Copy
            //value = op::Array<float>(shape);
            //memcpy(value.getPtr(), info.ptr, value.getVolume()*sizeof(float));

            return true;
        }

        // Cast op::Array<float> to numpy
        static handle cast(const op::Array<float> &m, return_value_policy, handle defval)
        {
            std::string format = format_descriptor<float>::format();
            return array(buffer_info(
                m.getPseudoConstPtr(),/* Pointer to buffer */
                sizeof(float),        /* Size of one scalar */
                format,               /* Python struct-style format descriptor */
                m.getSize().size(),   /* Number of dimensions */
                m.getSize(),          /* Buffer dimensions */
                m.getStride()         /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail

// Numpy - cv::Mat interop
namespace pybind11 { namespace detail {

template <> struct type_caster<cv::Mat> {
    public:

        PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

        // Cast numpy to cv::Mat
        bool load(handle src, bool)
        {
            /* Try a default converting into a Python */
            //array b(src, true);
            array b = reinterpret_borrow<array>(src);
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

            std::vector<int> shape = {(int)info.shape[0], (int)info.shape[1]};

            value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
            return true;
        }

        // Cast cv::Mat to numpy
        static handle cast(const cv::Mat &m, return_value_policy, handle defval)
        {
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

#endif
