#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/datum.hpp>

namespace op
{
    Datum::Datum()
    {
    }

    // Copy constructor
    Datum::Datum(const Datum& datum) :
        // ID
        id{datum.id},
        name{datum.name},
        // Input image and rendered version
        cvInputData{datum.cvInputData},
        inputNetData{datum.inputNetData},
        outputData{datum.outputData},
        cvOutputData{datum.cvOutputData},
        // Resulting Array<float> data
        poseKeypoints{datum.poseKeypoints},
        poseHeatMaps{datum.poseHeatMaps},
        faceRectangles{datum.faceRectangles},
        faceKeypoints{datum.faceKeypoints},
        handRectangles{datum.handRectangles},
        handKeypoints(datum.handKeypoints), // Parentheses instead of braces to avoid error in GCC 4.8
        // Other parameters
        scaleInputToOutput{datum.scaleInputToOutput},
        scaleNetToOutput{datum.scaleNetToOutput},
        scaleRatios{datum.scaleRatios},
        elementRendered{datum.elementRendered}
    {
    }

    // Copy assignment
    Datum& Datum::operator=(const Datum& datum)
    {
        try
        {
            // ID
            id = datum.id;
            name = datum.name;
            // Input image and rendered version
            cvInputData = datum.cvInputData;
            inputNetData = datum.inputNetData;
            outputData = datum.outputData;
            cvOutputData = datum.cvOutputData;
            // Resulting Array<float> data
            poseKeypoints = datum.poseKeypoints;
            poseHeatMaps = datum.poseHeatMaps,
            faceRectangles = datum.faceRectangles,
            faceKeypoints = datum.faceKeypoints,
            handRectangles = datum.handRectangles,
            handKeypoints = datum.handKeypoints,
            // Other parameters
            scaleInputToOutput = datum.scaleInputToOutput;
            scaleNetToOutput = datum.scaleNetToOutput;
            scaleRatios = datum.scaleRatios;
            elementRendered = datum.elementRendered;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    // Move constructor
    Datum::Datum(Datum&& datum) :
        // ID
        id{datum.id},
        // Other parameters
        scaleInputToOutput{datum.scaleInputToOutput},
        scaleNetToOutput{datum.scaleNetToOutput}
    {
        try
        {
            // ID
            std::swap(name, datum.name);
            // Input image and rendered version
            std::swap(cvInputData, datum.cvInputData);
            std::swap(inputNetData, datum.inputNetData);
            std::swap(outputData, datum.outputData);
            std::swap(cvOutputData, datum.cvOutputData);
            // Resulting Array<float> data
            std::swap(poseKeypoints, datum.poseKeypoints);
            std::swap(poseHeatMaps, datum.poseHeatMaps);
            std::swap(faceRectangles, datum.faceRectangles);
            std::swap(faceKeypoints, datum.faceKeypoints);
            std::swap(handRectangles, datum.handRectangles);
            std::swap(handKeypoints, datum.handKeypoints);
            // Other parameters
            std::swap(scaleRatios, datum.scaleRatios);
            std::swap(elementRendered, datum.elementRendered);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // Move assignment
    Datum& Datum::operator=(Datum&& datum)
    {
        try
        {
            // ID
            id = datum.id;
            std::swap(name, datum.name);
            // Input image and rendered version
            std::swap(cvInputData, datum.cvInputData);
            std::swap(inputNetData, datum.inputNetData);
            std::swap(outputData, datum.outputData);
            std::swap(cvOutputData, datum.cvOutputData);
            // Resulting Array<float> data
            std::swap(poseKeypoints, datum.poseKeypoints);
            std::swap(poseHeatMaps, datum.poseHeatMaps);
            std::swap(faceRectangles, datum.faceRectangles);
            std::swap(faceKeypoints, datum.faceKeypoints);
            std::swap(handRectangles, datum.handRectangles);
            std::swap(handKeypoints, datum.handKeypoints);
            // Other parameters
            scaleInputToOutput = datum.scaleInputToOutput;
            scaleNetToOutput = datum.scaleNetToOutput;
            std::swap(scaleRatios, datum.scaleRatios);
            std::swap(elementRendered, datum.elementRendered);
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    Datum::~Datum()
    {
    }

    Datum Datum::clone() const
    {
        try
        {
            // Constructor
            Datum datum;
            // ID
            datum.id = id;
            datum.name = name;
            // Input image and rendered version
            datum.cvInputData = cvInputData.clone();
            datum.inputNetData = inputNetData.clone();
            datum.outputData = outputData.clone();
            datum.cvOutputData = cvOutputData.clone();
            // Resulting Array<float> data
            datum.poseKeypoints = poseKeypoints.clone();
            datum.poseHeatMaps = poseHeatMaps.clone();
            datum.faceRectangles = faceRectangles;
            datum.faceKeypoints = faceKeypoints.clone();
            datum.handRectangles = datum.handRectangles;
            datum.handKeypoints[0] = handKeypoints[0].clone();
            datum.handKeypoints[1] = handKeypoints[1].clone();
            // Other parameters
            datum.scaleInputToOutput = scaleInputToOutput;
            datum.scaleNetToOutput = scaleNetToOutput;
            datum.scaleRatios = scaleRatios;
            datum.elementRendered = elementRendered;
            // Return
            return std::move(datum);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Datum{};
        }
    }
}
