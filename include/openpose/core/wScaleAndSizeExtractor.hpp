#ifndef OPENPOSE_CORE_W_SCALE_AND_SIZE_EXTRACTOR_HPP
#define OPENPOSE_CORE_W_SCALE_AND_SIZE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/scaleAndSizeExtractor.hpp>
#include <openpose/thread/worker.hpp>

namespace op
{
    template<typename TDatums>
    class WScaleAndSizeExtractor : public Worker<TDatums>
    {
    public:
        explicit WScaleAndSizeExtractor(const std::shared_ptr<ScaleAndSizeExtractor>& scaleAndSizeExtractor);

        virtual ~WScaleAndSizeExtractor();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<ScaleAndSizeExtractor> spScaleAndSizeExtractor;

        DELETE_COPY(WScaleAndSizeExtractor);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WScaleAndSizeExtractor<TDatums>::WScaleAndSizeExtractor(
        const std::shared_ptr<ScaleAndSizeExtractor>& scaleAndSizeExtractor) :
        spScaleAndSizeExtractor{scaleAndSizeExtractor}
    {
    }

    template<typename TDatums>
    WScaleAndSizeExtractor<TDatums>::~WScaleAndSizeExtractor()
    {
    }

    template<typename TDatums>
    void WScaleAndSizeExtractor<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WScaleAndSizeExtractor<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // cv::Mat -> float*
                for (auto& tDatumPtr : *tDatums)
                {
                    const Point<int> inputSize{tDatumPtr->cvInputData.cols, tDatumPtr->cvInputData.rows};
                    std::tie(tDatumPtr->scaleInputToNetInputs, tDatumPtr->netInputSizes, tDatumPtr->scaleInputToOutput,
                        tDatumPtr->netOutputSize) = spScaleAndSizeExtractor->extract(inputSize);
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WScaleAndSizeExtractor);
}

#endif // OPENPOSE_CORE_W_SCALE_AND_SIZE_EXTRACTOR_HPP
