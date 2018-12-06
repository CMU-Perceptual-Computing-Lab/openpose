#include <openpose/core/renderer.hpp>

namespace op
{
    Renderer::Renderer(const float renderThreshold, const float alphaKeypoint,
                       const float alphaHeatMap, const bool blendOriginalFrame,
                       const unsigned int elementToRender, const unsigned int numberElementsToRender) :
        mRenderThreshold{renderThreshold},
        mBlendOriginalFrame{blendOriginalFrame},
        spElementToRender{std::make_shared<std::atomic<unsigned int>>(elementToRender)},
        spNumberElementsToRender{std::make_shared<const unsigned int>(numberElementsToRender)},
        mShowGooglyEyes{false},
        mAlphaKeypoint{alphaKeypoint},
        mAlphaHeatMap{alphaHeatMap}
    {
    }

    Renderer::~Renderer()
    {
    }

    void Renderer::increaseElementToRender(const int increment)
    {
        try
        {
            // Sanity check
            if (*spNumberElementsToRender == 0)
                error("Number elements to render cannot be 0 for this function.",
                      __LINE__, __FUNCTION__, __FILE__);
            // Get current element to be rendered
            auto elementToRender = (((int)(*spElementToRender) + increment) % (int)(*spNumberElementsToRender));
            // Handling negative increments
            while (elementToRender < 0)
                elementToRender += *spNumberElementsToRender;
            // Update final value
            *spElementToRender = elementToRender;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Renderer::setElementToRender(const int elementToRender)
    {
        try
        {
            *spElementToRender = elementToRender % *spNumberElementsToRender;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void Renderer::setElementToRender(const ElementToRender elementToRender)
    {
        try
        {
            if (elementToRender == ElementToRender::Skeleton)
                *spElementToRender = 0;
            else if (elementToRender == ElementToRender::Background)
                *spElementToRender = 1;
            else if (elementToRender == ElementToRender::AddKeypoints)
                *spElementToRender = 2;
            else if (elementToRender == ElementToRender::AddPAFs)
                *spElementToRender = 3;
            else
                error("Unknown ElementToRender value.", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    float Renderer::getAlphaKeypoint() const
    {
        try
        {
            return mAlphaKeypoint;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    void Renderer::setAlphaKeypoint(const float alphaKeypoint)
    {
        try
        {
            mAlphaKeypoint = alphaKeypoint;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    float Renderer::getAlphaHeatMap() const
    {
        try
        {
            return mAlphaHeatMap;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    void Renderer::setAlphaHeatMap(const float alphaHeatMap)
    {
        try
        {
            mAlphaHeatMap = alphaHeatMap;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool Renderer::getBlendOriginalFrame() const
    {
        try
        {
            return mBlendOriginalFrame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void Renderer::setBlendOriginalFrame(const bool blendOriginalFrame)
    {
        try
        {
            mBlendOriginalFrame = blendOriginalFrame;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    bool Renderer::getShowGooglyEyes() const
    {
        try
        {
            return mShowGooglyEyes;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void Renderer::setShowGooglyEyes(const bool showGooglyEyes)
    {
        try
        {
            mShowGooglyEyes = showGooglyEyes;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
