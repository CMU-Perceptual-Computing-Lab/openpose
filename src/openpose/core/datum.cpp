#include <openpose/core/datum.hpp>
#include <openpose/utilities/errorAndLog.hpp>

namespace op
{
    Datum::Datum() :
        id{std::numeric_limits<unsigned long long>::max()},
        subId{0},
        subIdMax{0},
        poseIds{-1}
    {
    }

    // Copy constructor
    Datum::Datum(const Datum& datum) :
        // ID
        id{datum.id},
        subId{datum.subId},
        subIdMax{datum.subIdMax},
        name{datum.name},
        frameNumber{datum.frameNumber},
        // Input image and rendered version
        cvInputData{datum.cvInputData},
        inputNetData{datum.inputNetData},
        outputData{datum.outputData},
        cvOutputData{datum.cvOutputData},
        // Resulting Array<float> data parameters
        poseKeypoints{datum.poseKeypoints},
        poseIds{datum.poseIds},
        poseScores{datum.poseScores},
        poseHeatMaps{datum.poseHeatMaps},
        poseCandidates{datum.poseCandidates},
        faceRectangles{datum.faceRectangles},
        faceKeypoints{datum.faceKeypoints},
        faceHeatMaps{datum.faceHeatMaps},
        handRectangles{datum.handRectangles},
        handKeypoints(datum.handKeypoints), // Parentheses instead of braces to avoid error in GCC 4.8
        handHeatMaps(datum.handHeatMaps), // Parentheses instead of braces to avoid error in GCC 4.8
        // 3-D Reconstruction parameters
        poseKeypoints3D{datum.poseKeypoints3D},
        faceKeypoints3D{datum.faceKeypoints3D},
        handKeypoints3D(datum.handKeypoints3D), // Parentheses instead of braces to avoid error in GCC 4.8
        cameraMatrix{datum.cameraMatrix},
        // Other parameters
        scaleInputToNetInputs{datum.scaleInputToNetInputs},
        netInputSizes{datum.netInputSizes},
        scaleInputToOutput{datum.scaleInputToOutput},
        scaleNetToOutput{datum.scaleNetToOutput},
        elementRendered{datum.elementRendered}
        // 3D/Adam parameters
        #ifdef USE_3D_ADAM_MODEL
            ,
            // Adam/Unity params
            adamPosePtr{datum.adamPosePtr},
            adamPoseRows{datum.adamPoseRows},
            adamTranslationPtr{datum.adamTranslationPtr},
            // Adam params (Jacobians)
            vtVecPtr{datum.vtVecPtr},
            vtVecRows{datum.vtVecRows},
            j0VecPtr{datum.j0VecPtr},
            j0VecRows{datum.j0VecRows},
            adamFaceCoeffsExpPtr{datum.adamFaceCoeffsExpPtr},
            adamFaceCoeffsExpRows{datum.adamFaceCoeffsExpRows}
            #ifdef USE_EIGEN
                ,
                // Adam/Unity params
                adamPose{datum.adamPose},
                adamTranslation{datum.adamTranslation},
                // Adam params (Jacobians)
                vtVec{datum.vtVec},
                j0Vec{datum.j0Vec},
                adamFaceCoeffsExp{datum.adamFaceCoeffsExp}
            #endif
        #endif
    {
    }

    // Copy assignment
    Datum& Datum::operator=(const Datum& datum)
    {
        try
        {
            // ID
            id = datum.id;
            subId = datum.subId;
            subIdMax = datum.subIdMax;
            name = datum.name;
            frameNumber = datum.frameNumber;
            // Input image and rendered version
            cvInputData = datum.cvInputData;
            inputNetData = datum.inputNetData;
            outputData = datum.outputData;
            cvOutputData = datum.cvOutputData;
            // Resulting Array<float> data parameters
            poseKeypoints = datum.poseKeypoints;
            poseIds = datum.poseIds,
            poseScores = datum.poseScores,
            poseHeatMaps = datum.poseHeatMaps,
            poseCandidates = datum.poseCandidates,
            faceRectangles = datum.faceRectangles,
            faceKeypoints = datum.faceKeypoints,
            faceHeatMaps = datum.faceHeatMaps,
            handRectangles = datum.handRectangles,
            handKeypoints = datum.handKeypoints,
            handHeatMaps = datum.handHeatMaps,
            // 3-D Reconstruction parameters
            poseKeypoints3D = datum.poseKeypoints3D,
            faceKeypoints3D = datum.faceKeypoints3D,
            handKeypoints3D = datum.handKeypoints3D,
            cameraMatrix = datum.cameraMatrix;
            // Other parameters
            scaleInputToNetInputs = datum.scaleInputToNetInputs;
            netInputSizes = datum.netInputSizes;
            scaleInputToOutput = datum.scaleInputToOutput;
            scaleNetToOutput = datum.scaleNetToOutput;
            elementRendered = datum.elementRendered;
            // 3D/Adam parameters
            #ifdef USE_3D_ADAM_MODEL
                // Adam/Unity params
                adamPosePtr = datum.adamPosePtr;
                adamPoseRows = datum.adamPoseRows;
                adamTranslationPtr = datum.adamTranslationPtr;
                // Adam params (Jacobians)
                vtVecPtr = datum.vtVecPtr;
                vtVecRows = datum.vtVecRows;
                j0VecPtr = datum.j0VecPtr;
                j0VecRows = datum.j0VecRows;
                adamFaceCoeffsExpPtr = datum.adamFaceCoeffsExpPtr;
                adamFaceCoeffsExpRows = datum.adamFaceCoeffsExpRows;
                #ifdef USE_EIGEN
                    // Adam/Unity params
                    adamPose = datum.adamPose;
                    adamTranslation = datum.adamTranslation;
                    // Adam params (Jacobians)
                    vtVec = datum.vtVec;
                    j0Vec = datum.j0Vec;
                    adamFaceCoeffsExp = datum.adamFaceCoeffsExp;
                #endif
            #endif
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
        subId{datum.subId},
        subIdMax{datum.subIdMax},
        frameNumber{datum.frameNumber},
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
            // Resulting Array<float> data parameters
            std::swap(poseKeypoints, datum.poseKeypoints);
            std::swap(poseIds, datum.poseIds);
            std::swap(poseScores, datum.poseScores);
            std::swap(poseHeatMaps, datum.poseHeatMaps);
            std::swap(poseCandidates, datum.poseCandidates);
            std::swap(faceRectangles, datum.faceRectangles);
            std::swap(faceKeypoints, datum.faceKeypoints);
            std::swap(faceHeatMaps, datum.faceHeatMaps);
            std::swap(handRectangles, datum.handRectangles);
            std::swap(handKeypoints, datum.handKeypoints);
            std::swap(handHeatMaps, datum.handHeatMaps);
            // 3-D Reconstruction parameters
            std::swap(poseKeypoints3D, datum.poseKeypoints3D);
            std::swap(faceKeypoints3D, datum.faceKeypoints3D);
            std::swap(handKeypoints3D, datum.handKeypoints3D);
            std::swap(cameraMatrix, datum.cameraMatrix);
            // Other parameters
            std::swap(scaleInputToNetInputs, datum.scaleInputToNetInputs);
            std::swap(netInputSizes, datum.netInputSizes);
            std::swap(elementRendered, datum.elementRendered);
            // 3D/Adam parameters
            #ifdef USE_3D_ADAM_MODEL
                // Adam/Unity params
                std::swap(adamPosePtr, datum.adamPosePtr);
                adamPoseRows = datum.adamPoseRows;
                std::swap(adamTranslationPtr, datum.adamTranslationPtr);
                // Adam params (Jacobians)
                std::swap(vtVecPtr, datum.vtVecPtr);
                vtVecRows = datum.vtVecRows;
                std::swap(j0VecPtr, datum.j0VecPtr);
                j0VecRows = datum.j0VecRows;
                std::swap(adamFaceCoeffsExpPtr, datum.adamFaceCoeffsExpPtr);
                adamFaceCoeffsExpRows = datum.adamFaceCoeffsExpRows;
                #ifdef USE_EIGEN
                    // Adam/Unity params
                    std::swap(adamPose, datum.adamPose);
                    std::swap(adamTranslation, datum.adamTranslation);
                    // Adam params (Jacobians)
                    std::swap(vtVec, datum.vtVec);
                    std::swap(j0Vec, datum.j0Vec);
                    std::swap(adamFaceCoeffsExp, datum.adamFaceCoeffsExp);
                #endif
            #endif
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
            subId = datum.subId;
            subIdMax = datum.subIdMax;
            std::swap(name, datum.name);
            frameNumber = datum.frameNumber;
            // Input image and rendered version
            std::swap(cvInputData, datum.cvInputData);
            std::swap(inputNetData, datum.inputNetData);
            std::swap(outputData, datum.outputData);
            std::swap(cvOutputData, datum.cvOutputData);
            // Resulting Array<float> data parameters
            std::swap(poseKeypoints, datum.poseKeypoints);
            std::swap(poseIds, datum.poseIds);
            std::swap(poseScores, datum.poseScores);
            std::swap(poseHeatMaps, datum.poseHeatMaps);
            std::swap(poseCandidates, datum.poseCandidates);
            std::swap(faceRectangles, datum.faceRectangles);
            std::swap(faceKeypoints, datum.faceKeypoints);
            std::swap(faceHeatMaps, datum.faceHeatMaps);
            std::swap(handRectangles, datum.handRectangles);
            std::swap(handKeypoints, datum.handKeypoints);
            std::swap(handHeatMaps, datum.handHeatMaps);
            // 3-D Reconstruction parameters
            std::swap(poseKeypoints3D, datum.poseKeypoints3D);
            std::swap(faceKeypoints3D, datum.faceKeypoints3D);
            std::swap(handKeypoints3D, datum.handKeypoints3D);
            std::swap(cameraMatrix, datum.cameraMatrix);
            // Other parameters
            std::swap(scaleInputToNetInputs, datum.scaleInputToNetInputs);
            std::swap(netInputSizes, datum.netInputSizes);
            std::swap(elementRendered, datum.elementRendered);
            // 3D/Adam parameters
            #ifdef USE_3D_ADAM_MODEL
                // Adam/Unity params
                std::swap(adamPosePtr, datum.adamPosePtr);
                adamPoseRows = datum.adamPoseRows;
                std::swap(adamTranslationPtr, datum.adamTranslationPtr);
                // Adam params (Jacobians)
                std::swap(vtVecPtr, datum.vtVecPtr);
                vtVecRows = datum.vtVecRows;
                std::swap(j0VecPtr, datum.j0VecPtr);
                j0VecRows = datum.j0VecRows;
                std::swap(adamFaceCoeffsExpPtr, datum.adamFaceCoeffsExpPtr);
                adamFaceCoeffsExpRows = datum.adamFaceCoeffsExpRows;
                #ifdef USE_EIGEN
                    // Adam/Unity params
                    std::swap(adamPose, datum.adamPose);
                    std::swap(adamTranslation, datum.adamTranslation);
                    // Adam params (Jacobians)
                    std::swap(vtVec, datum.vtVec);
                    std::swap(j0Vec, datum.j0Vec);
                    std::swap(adamFaceCoeffsExp, datum.adamFaceCoeffsExp);
                #endif
            #endif
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
            datum.subId = subId;
            datum.subIdMax = subIdMax;
            datum.name = name;
            datum.frameNumber = frameNumber;
            // Input image and rendered version
            datum.cvInputData = cvInputData.clone();
            datum.inputNetData.resize(inputNetData.size());
            for (auto i = 0u ; i < datum.inputNetData.size() ; i++)
                datum.inputNetData[i] = inputNetData[i].clone();
            datum.outputData = outputData.clone();
            datum.cvOutputData = cvOutputData.clone();
            // Resulting Array<float> data parameters
            datum.poseKeypoints = poseKeypoints.clone();
            datum.poseIds = poseIds.clone();
            datum.poseScores = poseScores.clone();
            datum.poseHeatMaps = poseHeatMaps.clone();
            datum.poseCandidates = poseCandidates;
            datum.faceRectangles = faceRectangles;
            datum.faceKeypoints = faceKeypoints.clone();
            datum.faceHeatMaps = faceHeatMaps.clone();
            datum.handRectangles = handRectangles;
            for (auto i = 0u ; i < datum.handKeypoints.size() ; i++)
                datum.handKeypoints[i] = handKeypoints[i].clone();
            for (auto i = 0u ; i < datum.handKeypoints.size() ; i++)
                datum.handHeatMaps[i] = handHeatMaps[i].clone();
            // 3-D Reconstruction parameters
            datum.poseKeypoints3D = poseKeypoints3D.clone();
            datum.faceKeypoints3D = faceKeypoints3D.clone();
            for (auto i = 0u ; i < datum.handKeypoints.size() ; i++)
                datum.handKeypoints3D[i] = handKeypoints3D[i].clone();
            datum.cameraMatrix = cameraMatrix.clone();
            // Other parameters
            datum.scaleInputToNetInputs = scaleInputToNetInputs;
            datum.netInputSizes = netInputSizes;
            datum.scaleInputToOutput = scaleInputToOutput;
            datum.scaleNetToOutput = scaleNetToOutput;
            datum.elementRendered = elementRendered;
            // 3D/Adam parameters
            #ifdef USE_3D_ADAM_MODEL
                // Adam/Unity params
                datum.adamPosePtr = adamPosePtr;
                datum.adamPoseRows = adamPoseRows;
                datum.adamTranslationPtr = adamTranslationPtr;
                // Adam params (Jacobians)
                datum.vtVecPtr = vtVecPtr;
                datum.vtVecRows = vtVecRows;
                datum.j0VecPtr = j0VecPtr;
                datum.j0VecRows = j0VecRows;
                datum.adamFaceCoeffsExpPtr = adamFaceCoeffsExpPtr;
                datum.adamFaceCoeffsExpRows = adamFaceCoeffsExpRows;
                #ifdef USE_EIGEN
                    // Adam/Unity params
                    datum.adamPose = adamPose;
                    datum.adamTranslation = adamTranslation;
                    // Adam params (Jacobians)
                    datum.vtVec = vtVec;
                    datum.j0Vec = j0Vec;
                    datum.adamFaceCoeffsExp = adamFaceCoeffsExp;
                #endif
            #endif
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
