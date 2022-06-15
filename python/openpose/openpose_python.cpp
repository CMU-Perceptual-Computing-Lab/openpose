#ifndef OPENPOSE_PYTHON_HPP
#define OPENPOSE_PYTHON_HPP
#define BOOST_DATE_TIME_NO_LIB

#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <stdexcept>

PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<op::Datum>>);

#ifdef _WIN32
    #define OP_EXPORT __declspec(dllexport)
#else
    #define OP_EXPORT
#endif

namespace op
{

    namespace py = pybind11;

    void parse_gflags(const std::vector<std::string>& argv)
    {
        try
        {
            std::vector<char*> argv_vec;
            for (auto& arg : argv)
                argv_vec.emplace_back((char*)arg.c_str());
            char** cast = &argv_vec[0];
            int size = (int)argv_vec.size();
            gflags::ParseCommandLineFlags(&size, &cast, true);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void init_int(py::dict d)
    {
        try
        {
            std::vector<std::string> argv;
            argv.emplace_back("openpose.py");
            for (auto item : d){
                // Sanity check
                std::size_t found = std::string(py::str(item.first)).find("=");
                if (found != std::string::npos)
                    error("PyOpenPose does not support equal sign flags (e.g., "
                        + std::string(py::str(item.first)) + ").", __LINE__, __FUNCTION__, __FILE__);
                // Add argument
                argv.emplace_back("--" + std::string(py::str(item.first)) + "=" + std::string(py::str(item.second)));
            }
            parse_gflags(argv);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void init_argv(std::vector<std::string> argv)
    {
        try
        {
            argv.insert(argv.begin(), "openpose.py");
            parse_gflags(argv);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    class WrapperPython{
    public:
        std::unique_ptr<Wrapper> opWrapper;
        bool synchronousIn;

        WrapperPython(ThreadManagerMode mode = ThreadManagerMode::Asynchronous)
        {
            opLog("Starting OpenPose Python Wrapper...", Priority::High);

            // Construct opWrapper
            opWrapper = std::unique_ptr<Wrapper>(new Wrapper(mode));

            // Synchronous in
            synchronousIn = (
                mode == ThreadManagerMode::AsynchronousOut ||
                mode == ThreadManagerMode::Synchronous
            );
        }

        void configure(py::dict params = py::dict())
        {
            try
            {
                if (params.size())
                    init_int(params);

                // logging_level
                checkBool(
                    0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                    __LINE__, __FUNCTION__, __FILE__);
                ConfigureLog::setPriorityThreshold((Priority)FLAGS_logging_level);
                Profiler::setDefaultX(FLAGS_profile_speed);

                // Applying user defined configuration - GFlags to program variables
                // outputSize
                const auto outputSize = flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
                // netInputSize
                const auto netInputSize = flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
                // faceNetInputSize
                const auto faceNetInputSize = flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
                // handNetInputSize
                const auto handNetInputSize = flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
                // poseMode
                const auto poseMode = flagsToPoseMode(FLAGS_body);
                // poseModel
                const auto poseModel = flagsToPoseModel(op::String(FLAGS_model_pose));
                // JSON saving
                if (!FLAGS_write_keypoint.empty())
                    opLog("Flag `write_keypoint` is deprecated and will eventually be removed."
                            " Please, use `write_json` instead.", Priority::Max);
                // keypointScaleMode
                const auto keypointScaleMode = flagsToScaleMode(FLAGS_keypoint_scale);
                // heatmaps to add
                const auto heatMapTypes = flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                              FLAGS_heatmaps_add_PAFs);
                const auto heatMapScaleMode = flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
                // >1 camera view?
                const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
                // Face and hand detectors
                const auto faceDetector = flagsToDetector(FLAGS_face_detector);
                const auto handDetector = flagsToDetector(FLAGS_hand_detector);
                // Enabling Google Logging
                const bool enableGoogleLogging = true;

                // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
                const op::WrapperStructPose wrapperStructPose{
                    poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
                    FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
                    op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
                    (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
                    heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
                    FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
                    op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
                opWrapper->configure(wrapperStructPose);
                // Face configuration (use WrapperStructFace{} to disable it)
                const WrapperStructFace wrapperStructFace{
                    FLAGS_face, faceDetector, faceNetInputSize,
                    flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
                    (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
                opWrapper->configure(wrapperStructFace);
                // Hand configuration (use WrapperStructHand{} to disable it)
                const WrapperStructHand wrapperStructHand{
                    FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
                    flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
                    (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
                opWrapper->configure(wrapperStructHand);
                // Extra functionality configuration (use WrapperStructExtra{} to disable it)
                const WrapperStructExtra wrapperStructExtra{
                    FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
                opWrapper->configure(wrapperStructExtra);
                // Output (comment or use default argument to disable any output)
                const WrapperStructOutput wrapperStructOutput{
                    FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
                    op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
                    FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
                    op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
                    op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
                    op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
                    op::String(FLAGS_udp_port)};
                opWrapper->configure(wrapperStructOutput);
                if (synchronousIn) {
                    // SynchronousIn => We need a producer

                    // Producer (use default to disable any input)
                    const auto cameraSize = flagsToPoint(op::String(FLAGS_camera_resolution), "-1x-1");
                    ProducerType producerType;
                    op::String producerString;
                    std::tie(producerType, producerString) = flagsToProducer(
                        op::String(FLAGS_image_dir), op::String(FLAGS_video), op::String(FLAGS_ip_camera), FLAGS_camera,
                        FLAGS_flir_camera, FLAGS_flir_camera_index);
                    const WrapperStructInput wrapperStructInput{
                        producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
                        FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
                        cameraSize, op::String(FLAGS_camera_parameter_path), FLAGS_frame_undistort, FLAGS_3d_views};
                    opWrapper->configure(wrapperStructInput);
                }
                // No GUI. Equivalent to: opWrapper.configure(WrapperStructGui{});
                // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
                if (FLAGS_disable_multi_thread)
                    opWrapper->disableMultiThreading();
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void start()
        {
            try
            {
                opWrapper->start();
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void stop()
        {
            try
            {
                opWrapper->stop();
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void exec()
        {
            try
            {
                // GUI (comment or use default argument to disable any visual output)
                const WrapperStructGui wrapperStructGui{
                    flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
                opWrapper->configure(wrapperStructGui);
                opWrapper->exec();
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        bool emplaceAndPop(std::vector<std::shared_ptr<Datum>>& l)
        {
            try
            {
                std::shared_ptr<std::vector<std::shared_ptr<Datum>>> datumsPtr(
                    &l,
                    [](std::vector<std::shared_ptr<Datum>>*){}
                );
                auto got = opWrapper->emplaceAndPop(datumsPtr);
                if (got && datumsPtr.get() != &l) {
                    l.swap(*datumsPtr);
                }
                return got;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return false;
            }
        }

        bool waitAndEmplace(std::vector<std::shared_ptr<Datum>>& l)
        {
            try
            {
                std::shared_ptr<std::vector<std::shared_ptr<Datum>>> datumsPtr(&l);
                return opWrapper->waitAndEmplace(datumsPtr);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return false;
            }
        }

        bool waitAndPop(std::vector<std::shared_ptr<Datum>>& l)
        {
            try
            {
                std::shared_ptr<std::vector<std::shared_ptr<Datum>>> datumsPtr;
                auto got = opWrapper->waitAndPop(datumsPtr);
                if (got) {
                    l.swap(*datumsPtr);
                }
                return got;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return false;
            }
        }
    };

    std::vector<std::string> getImagesFromDirectory(const std::string& directoryPath)
    {
        try
        {
            return getFilesOnDirectory(directoryPath, Extensions::Images);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    PYBIND11_MODULE(pyopenpose, m) {

        // Functions for Init Params
        m.def("init_int", &init_int, "Init Function");
        m.def("init_argv", &init_argv, "Init Function");
        m.def("get_gpu_number", &getGpuNumber, "Get Total GPU");
        m.def("get_images_on_directory", &getImagesFromDirectory, "Get Images On Directory");

        // Pose Mapping
        // Code example in doc/02_output.md, section Keypoint Ordering in C++/Python
        m.def("getPoseBodyPartMapping", &getPoseBodyPartMapping, "getPoseBodyPartMapping");
        m.def("getPoseNumberBodyParts", &getPoseNumberBodyParts, "getPoseNumberBodyParts");
        m.def("getPosePartPairs", &getPosePartPairs, "getPosePartPairs");
        m.def("getPoseMapIndex", &getPoseMapIndex, "getPoseMapIndex");
        py::enum_<PoseModel>(m, "PoseModel", py::arithmetic())
                .value("BODY_25", PoseModel::BODY_25)
                .value("COCO_18", PoseModel::COCO_18)
                .value("MPI_15", PoseModel::MPI_15)
                .value("MPI_15_4", PoseModel::MPI_15_4)
                .value("BODY_25B", PoseModel::BODY_25B)
                .value("BODY_135", PoseModel::BODY_135)
                .export_values();

        // OpenposePython
        py::class_<WrapperPython>(m, "WrapperPython")
            .def(py::init<>())
            .def(py::init<ThreadManagerMode>())
            .def("configure", &WrapperPython::configure)
            .def("start", &WrapperPython::start)
            .def("stop", &WrapperPython::stop)
            .def("execute", &WrapperPython::exec)
            .def("emplaceAndPop", &WrapperPython::emplaceAndPop)
            .def("waitAndEmplace", &WrapperPython::waitAndEmplace)
            .def("waitAndPop", &WrapperPython::waitAndPop)
            ;

        // ThreadManagerMode
        py::enum_<ThreadManagerMode>(m, "ThreadManagerMode")
            .value("Asynchronous", ThreadManagerMode::Asynchronous)
            .value("AsynchronousIn", ThreadManagerMode::AsynchronousIn)
            .value("AsynchronousOut", ThreadManagerMode::AsynchronousOut)
            .value("Synchronous", ThreadManagerMode::Synchronous)
            ;

        // Datum Object
        py::class_<Datum, std::shared_ptr<Datum>>(m, "Datum")
            .def(py::init<>())
            .def_readwrite("id", &Datum::id)
            .def_readwrite("subId", &Datum::subId)
            .def_readwrite("subIdMax", &Datum::subIdMax)
            .def_readwrite("name", &Datum::name)
            .def_readwrite("frameNumber", &Datum::frameNumber)
            .def_readwrite("cvInputData", &Datum::cvInputData)
            .def_readwrite("inputNetData", &Datum::inputNetData)
            .def_readwrite("outputData", &Datum::outputData)
            .def_readwrite("cvOutputData", &Datum::cvOutputData)
            .def_readwrite("cvOutputData3D", &Datum::cvOutputData3D)
            .def_readwrite("poseKeypoints", &Datum::poseKeypoints)
            .def_readwrite("poseIds", &Datum::poseIds)
            .def_readwrite("poseScores", &Datum::poseScores)
            .def_readwrite("poseHeatMaps", &Datum::poseHeatMaps)
            .def_readwrite("poseCandidates", &Datum::poseCandidates)
            .def_readwrite("faceRectangles", &Datum::faceRectangles)
            .def_readwrite("faceKeypoints", &Datum::faceKeypoints)
            .def_readwrite("faceHeatMaps", &Datum::faceHeatMaps)
            .def_readwrite("handRectangles", &Datum::handRectangles)
            .def_readwrite("handKeypoints", &Datum::handKeypoints)
            .def_readwrite("handHeatMaps", &Datum::handHeatMaps)
            .def_readwrite("poseKeypoints3D", &Datum::poseKeypoints3D)
            .def_readwrite("faceKeypoints3D", &Datum::faceKeypoints3D)
            .def_readwrite("handKeypoints3D", &Datum::handKeypoints3D)
            .def_readwrite("cameraMatrix", &Datum::cameraMatrix)
            .def_readwrite("cameraExtrinsics", &Datum::cameraExtrinsics)
            .def_readwrite("cameraIntrinsics", &Datum::cameraIntrinsics)
            .def_readwrite("poseNetOutput", &Datum::poseNetOutput)
            .def_readwrite("scaleInputToNetInputs", &Datum::scaleInputToNetInputs)
            .def_readwrite("netInputSizes", &Datum::netInputSizes)
            .def_readwrite("scaleInputToOutput", &Datum::scaleInputToOutput)
            .def_readwrite("netOutputSize", &Datum::netOutputSize)
            .def_readwrite("scaleNetToOutput", &Datum::scaleNetToOutput)
            .def_readwrite("elementRendered", &Datum::elementRendered)
            ;

        py::bind_vector<std::vector<std::shared_ptr<Datum>>>(m, "VectorDatum");

        // Rectangle
        py::class_<Rectangle<float>>(m, "Rectangle")
            .def("__repr__", [](Rectangle<float> &a) { return a.toString(); })
            .def(py::init<>())
            .def(py::init<float, float, float, float>())
            .def_readwrite("x", &Rectangle<float>::x)
            .def_readwrite("y", &Rectangle<float>::y)
            .def_readwrite("width", &Rectangle<float>::width)
            .def_readwrite("height", &Rectangle<float>::height)
            ;

        // Point
        py::class_<Point<int>>(m, "Point")
            .def("__repr__", [](Point<int> &a) { return a.toString(); })
            .def(py::init<>())
            .def(py::init<int, int>())
            .def_readwrite("x", &Point<int>::x)
            .def_readwrite("y", &Point<int>::y)
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
            try
            {
                UNUSED(imp);
                // array b(src, true);
                array b = reinterpret_borrow<array>(src);
                buffer_info info = b.request();

                if (info.format != format_descriptor<float>::format())
                    op::error("op::Array only supports float32 now", __LINE__, __FUNCTION__, __FILE__);

                //std::vector<int> a(info.shape);
                std::vector<int> shape(std::begin(info.shape), std::end(info.shape));

                // No copy
                value = op::Array<float>(shape, (float*)info.ptr);
                // Copy
                //value = op::Array<float>(shape);
                //memcpy(value.getPtr(), info.ptr, value.getVolume()*sizeof(float));

                return true;
            }
            catch (const std::exception& e)
            {
                op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }

        // Cast op::Array<float> to numpy
        static handle cast(const op::Array<float> &m, return_value_policy, handle defval)
        {
            UNUSED(defval);
            if (m.getSize().size() == 0) {
                return Py_BuildValue("");
            }
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

// Numpy - op::Array<long long> interop
namespace pybind11 { namespace detail {

template <> struct type_caster<op::Array<long long>> {
    public:

        PYBIND11_TYPE_CASTER(op::Array<long long>, _("numpy.ndarray"));

        // Cast numpy to op::Array<long long>
        bool load(handle src, bool imp)
        {
            op::error("op::Array<long long> is read only now", __LINE__, __FUNCTION__, __FILE__);
            return false;
        }

        // Cast op::Array<long long> to numpy
        static handle cast(const op::Array<long long> &m, return_value_policy, handle defval)
        {
            UNUSED(defval);
            if (m.getSize().size() == 0) {
                return Py_BuildValue("");
            }
            std::string format = format_descriptor<long long>::format();
            return array(buffer_info(
                m.getPseudoConstPtr(),/* Pointer to buffer */
                sizeof(long long),    /* Size of one scalar */
                format,               /* Python struct-style format descriptor */
                m.getSize().size(),   /* Number of dimensions */
                m.getSize(),          /* Buffer dimensions */
                m.getStride()         /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail

// Numpy - op::Matrix interop
namespace pybind11 { namespace detail {

template <> struct type_caster<op::Matrix> {
    public:

        PYBIND11_TYPE_CASTER(op::Matrix, _("numpy.ndarray"));

        // Cast numpy to op::Matrix
        bool load(handle src, bool)
        {
            /* Try a default converting into a Python */
            //array b(src, true);
            array b = reinterpret_borrow<array>(src);
            buffer_info info = b.request();

            const int ndims = (int)info.ndim;

            decltype(CV_32F) dtype;
            size_t elemsize;
            if (info.format == format_descriptor<float>::format())
            {
                if (ndims == 3)
                    dtype = CV_32FC3;
                else
                    dtype = CV_32FC1;
                elemsize = sizeof(float);
            }
            else if (info.format == format_descriptor<double>::format())
            {
                if (ndims == 3)
                    dtype = CV_64FC3;
                else
                    dtype = CV_64FC1;
                elemsize = sizeof(double);
            }
            else if (info.format == format_descriptor<unsigned char>::format())
            {
                if (ndims == 3)
                    dtype = CV_8UC3;
                else
                    dtype = CV_8UC1;
                elemsize = sizeof(unsigned char);
            }
            else
            {
                throw std::logic_error("Unsupported type");
                return false;
            }

            std::vector<int> shape = {(int)info.shape[0], (int)info.shape[1]};

            value = op::Matrix(shape[0], shape[1], dtype, info.ptr);
            // value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
            return true;
        }

        // Cast op::Matrix to numpy
        static handle cast(const op::Matrix &matrix, return_value_policy, handle defval)
        {
            UNUSED(defval);
            std::string format = format_descriptor<unsigned char>::format();
            size_t elemsize = sizeof(unsigned char);
            int dim;
            switch(matrix.type()) {
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
                bufferdim = {(size_t) matrix.rows(), (size_t) matrix.cols()};
                strides = {elemsize * (size_t) matrix.cols(), elemsize};
            } else if (dim == 3) {
                bufferdim = {(size_t) matrix.rows(), (size_t) matrix.cols(), (size_t) 3};
                strides = {(size_t) elemsize * matrix.cols() * 3, (size_t) elemsize * 3, (size_t) elemsize};
            }
            return array(buffer_info(
                matrix.dataPseudoConst(),   /* Pointer to buffer */
                elemsize,                   /* Size of one scalar */
                format,                     /* Python struct-style format descriptor */
                dim,                        /* Number of dimensions */
                bufferdim,                  /* Buffer dimensions */
                strides                     /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail

#endif
