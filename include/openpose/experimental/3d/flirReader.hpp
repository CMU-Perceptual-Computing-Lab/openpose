#ifndef OPENPOSE_EXPERIMENTAL_3D_POINT_GREY_HPP
#define OPENPOSE_EXPERIMENTAL_3D_POINT_GREY_HPP

#include <openpose/core/common.hpp>
#include <openpose/thread/workerProducer.hpp>

namespace op
{
    /**
     * FLIR (Point-Grey) camera producer
     * Following OpenPose `tutorial_wrapper/` examples, we create our own class inherited from WorkerProducer.
     * This worker:
     * 1. Set hardware trigger and the buffer to get the latest obtained frame.
     * 2. Read images from FLIR cameras.
     * 3. Turn them into std::vector<cv::Mat>.
     * 4. Return the resulting images wrapped into a std::shared_ptr<std::vector<Datum>>.
     * The HW trigger + reading FLIR camera code is highly based on the Spinnaker SDK examples
     * `AcquisitionMultipleCamera` and specially `Trigger`
     * (located in `src/`). See them for more details about the cameras.
     * See `examples/tutorial_wrapper/` for more details about inhering the WorkerProducer class.
    */
    class OP_API WFlirReader : public WorkerProducer<std::shared_ptr<std::vector<Datum>>>
    {
    public:
        WFlirReader();

        ~WFlirReader();

        void initializationOnThread();

        std::shared_ptr<std::vector<Datum>> workProducer();

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplWFlirReader;
        std::unique_ptr<ImplWFlirReader> upImpl;
    };
}

#endif // OPENPOSE_EXPERIMENTAL_3D_POINT_GREY_HPP
