#ifndef OPENPOSE_CORE_RENDERER_HPP
#define OPENPOSE_CORE_RENDERER_HPP

#include <atomic>
#include <openpose/core/common.hpp>

namespace op
{
    class OP_API Renderer
    {
    public:
        explicit Renderer(const float renderThreshold, const float alphaKeypoint, const float alphaHeatMap,
                          const bool blendOriginalFrame = true, const unsigned int elementToRender = 0u,
                          const unsigned int numberElementsToRender = 0u);

        void increaseElementToRender(const int increment);

        void setElementToRender(const int elementToRender);

        bool getBlendOriginalFrame() const;

        void setBlendOriginalFrame(const bool blendOriginalFrame);

        float getAlphaKeypoint() const;

        void setAlphaKeypoint(const float alphaKeypoint);

        float getAlphaHeatMap() const;

        void setAlphaHeatMap(const float alphaHeatMap);

        bool getShowGooglyEyes() const;

        void setShowGooglyEyes(const bool showGooglyEyes);

    protected:
        const float mRenderThreshold;
        std::atomic<bool> mBlendOriginalFrame;
        std::shared_ptr<std::atomic<unsigned int>> spElementToRender;
        std::shared_ptr<const unsigned int> spNumberElementsToRender;
        std::atomic<bool> mShowGooglyEyes;

    private:
        float mAlphaKeypoint;
        float mAlphaHeatMap;

        DELETE_COPY(Renderer);
    };
}

#endif // OPENPOSE_CORE_RENDERER_HPP
