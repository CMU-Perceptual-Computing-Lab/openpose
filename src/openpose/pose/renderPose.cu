#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/cuda.hu>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/render.hu>
#include <openpose/pose/renderPose.hpp>

namespace op
{
    // PI digits: http://www.piday.org/million/
    __constant__ const float PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745f;
    // Keypoint pairs
    __constant__ const unsigned int BODY_25_PAIRS_GPU[] = {POSE_BODY_25_PAIRS_RENDER_GPU};
    __constant__ const unsigned int COCO_PAIRS_GPU[] = {POSE_COCO_PAIRS_RENDER_GPU};
    __constant__ const unsigned int BODY_19_PAIRS_GPU[] = {POSE_BODY_19_PAIRS_RENDER_GPU};
    __constant__ const unsigned int BODY_23_PAIRS_GPU[] = {POSE_BODY_23_PAIRS_RENDER_GPU};
    __constant__ const unsigned int BODY_59_PAIRS_GPU[] = {POSE_BODY_59_PAIRS_RENDER_GPU};
    __constant__ const unsigned int BODY_65_PAIRS_GPU[] = {POSE_BODY_65_PAIRS_RENDER_GPU};
    __constant__ const unsigned int MPI_PAIRS_GPU[] = {POSE_MPI_PAIRS_RENDER_GPU};
    __constant__ const unsigned int CAR_12_PAIRS_GPU[] = {POSE_CAR_12_PAIRS_RENDER_GPU};
    __constant__ const unsigned int CAR_22_PAIRS_GPU[] = {POSE_CAR_22_PAIRS_RENDER_GPU};
    // Keypoint scales
    __constant__ const float BODY_25_SCALES[] = {POSE_BODY_25_SCALES_RENDER_GPU};
    __constant__ const float COCO_SCALES[] = {POSE_COCO_SCALES_RENDER_GPU};
    __constant__ const float BODY_19_SCALES[] = {POSE_BODY_19_SCALES_RENDER_GPU};
    __constant__ const float BODY_23_SCALES[] = {POSE_BODY_23_SCALES_RENDER_GPU};
    __constant__ const float BODY_59_SCALES[] = {POSE_BODY_59_SCALES_RENDER_GPU};
    __constant__ const float BODY_65_SCALES[] = {POSE_BODY_65_SCALES_RENDER_GPU};
    __constant__ const float MPI_SCALES[] = {POSE_MPI_SCALES_RENDER_GPU};
    __constant__ const float CAR_12_SCALES[] = {POSE_CAR_12_SCALES_RENDER_GPU};
    __constant__ const float CAR_22_SCALES[] = {POSE_CAR_22_SCALES_RENDER_GPU};
    // RGB colors
    __constant__ const float BODY_25_COLORS[] = {POSE_BODY_25_COLORS_RENDER_GPU};
    __constant__ const float COCO_COLORS[] = {POSE_COCO_COLORS_RENDER_GPU};
    __constant__ const float BODY_19_COLORS[] = {POSE_BODY_19_COLORS_RENDER_GPU};
    __constant__ const float BODY_23_COLORS[] = {POSE_BODY_23_COLORS_RENDER_GPU};
    __constant__ const float BODY_59_COLORS[] = {POSE_BODY_59_COLORS_RENDER_GPU};
    __constant__ const float BODY_65_COLORS[] = {POSE_BODY_65_COLORS_RENDER_GPU};
    __constant__ const float MPI_COLORS[] = {POSE_MPI_COLORS_RENDER_GPU};
    __constant__ const float CAR_12_COLORS[] = {POSE_CAR_12_COLORS_RENDER_GPU};
    __constant__ const float CAR_22_COLORS[] = {POSE_CAR_22_COLORS_RENDER_GPU};



    inline __device__ void getColorHeatMap(float* colorPtr, float v, const float vmin, const float vmax)
    {
        v = fastTruncate(v, vmin, vmax);
        const auto dv = vmax - vmin;

        if (v < (vmin + 0.125f * dv))
        {
            colorPtr[0] = 256.f * (0.5f + (v * 4.f)); //B: 0.5 ~ 1
            colorPtr[1] = 0.f;
            colorPtr[2] = 0.f;
        }
        else if (v < (vmin + 0.375f * dv))
        {
            colorPtr[0] = 255.f;
            colorPtr[1] = 256.f * (v - 0.125f) * 4.f; //G: 0 ~ 1
            colorPtr[2] = 0.f;
        }
        else if (v < (vmin + 0.625f * dv))
        {
            colorPtr[0] = 256.f * (-4.f * v + 2.5f); //B: 1 ~ 0
            colorPtr[1] = 255.f;
            colorPtr[2] = 256.f * (4.f * (v - 0.375f)); // R: 0 ~ 1
        }
        else if (v < (vmin + 0.875f * dv))
        {
            colorPtr[0] = 0.f;
            colorPtr[1] = 256.f * (-4.f * v + 3.5f); //G: 1 ~ 0
            colorPtr[2] = 255.f;
        }
        else
        {
            colorPtr[0] = 0.f;
            colorPtr[1] = 0.f;
            colorPtr[2] = 256.f * (-4.f * v + 4.5f); //R: 1 ~ 0.5
        }
    }

    inline __device__ void getColorAffinity(float3& colorPtr, float v, const float vmin, const float vmax)
    {
        const auto RY = 15;
        const auto YG =  6;
        const auto GC =  4;
        const auto CB = 11;
        const auto BM = 13;
        const auto MR =  6;
        const auto summed = RY+YG+GC+CB+BM+MR;       // 55
        v = fastTruncate(v, vmin, vmax) * summed;

        if (v < RY)
            colorPtr = {255.f,                        255.f*(v/(RY)),               0.f};
        else if (v < RY+YG)
            colorPtr = {255.f*(1-((v-RY)/(YG))),      255.f,                        0.f};
        else if (v < RY+YG+GC)
            colorPtr = {0.f * (1-((v-RY)/(YG))),      255.f,                        255.f*((v-RY-YG)/(GC))};
        else if (v < RY+YG+GC+CB)
            colorPtr = {0.f,                          255.f*(1-((v-RY-YG-GC)/(CB))),255.f};
        else if (v < summed-MR)
            colorPtr = {255.f*((v-RY-YG-GC-CB)/(BM)), 0.f,                          255.f};
        else if (v < summed)
            colorPtr = {255.f,                        0.f,                          255.f*(1-((v-RY-YG-GC-CB-BM)/(MR)))};
        else
            colorPtr = {255.f,                        0.f,                          0.f};
    }

    inline __device__ void getColorXYAffinity(float3& colorPtr, const float x, const float y)
    {
        const auto rad = fastMin(1.f, sqrt( x*x + y*y ) );
        const float a = atan2(-y,-x)/PI;
        auto fk = (a+1.f)/2.f; // 0 to 1
        if (::isnan(fk))
            fk = 0.f;
        getColorAffinity(colorPtr, fk, 0.f, 1.f);
        colorPtr.x *= rad;
        colorPtr.y *= rad;
        colorPtr.z *= rad;
    }

    __global__ void renderPoseCoco(float* targetPtr, const int targetWidth, const int targetHeight,
                                   const float* const posePtr, const int numberPeople, const float threshold,
                                   const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(COCO_PAIRS_GPU) / (2*sizeof(COCO_PAIRS_GPU[0]));
        const auto numberScales = sizeof(COCO_SCALES) / sizeof(COCO_SCALES[0]);
        const auto numberColors = sizeof(COCO_COLORS) / (3*sizeof(COCO_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, COCO_PAIRS_GPU, numberPeople, 18, numberPartPairs, COCO_COLORS,
                        numberColors, radius, lineWidth, COCO_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 14 : -1), (googlyEyes ? 15 : -1));
    }

    __global__ void renderPoseBody19(float* targetPtr, const int targetWidth, const int targetHeight,
                                     const float* const posePtr, const int numberPeople, const float threshold,
                                     const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(BODY_19_PAIRS_GPU) / (2*sizeof(BODY_19_PAIRS_GPU[0]));
        const auto numberScales = sizeof(BODY_19_SCALES) / sizeof(BODY_19_SCALES[0]);
        const auto numberColors = sizeof(BODY_19_COLORS) / (3*sizeof(BODY_19_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, BODY_19_PAIRS_GPU, numberPeople, 19, numberPartPairs, BODY_19_COLORS, numberColors,
                        radius, lineWidth, BODY_19_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 15 : -1), (googlyEyes ? 16 : -1));
    }

    __global__ void renderPoseBody23(float* targetPtr, const int targetWidth, const int targetHeight,
                                     const float* const posePtr, const int numberPeople, const float threshold,
                                     const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(BODY_23_PAIRS_GPU) / (2*sizeof(BODY_23_PAIRS_GPU[0]));
        const auto numberScales = sizeof(BODY_23_SCALES) / sizeof(BODY_23_SCALES[0]);
        const auto numberColors = sizeof(BODY_23_COLORS) / (3*sizeof(BODY_23_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, BODY_23_PAIRS_GPU, numberPeople, 23, numberPartPairs, BODY_23_COLORS, numberColors,
                        radius, lineWidth, BODY_23_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 13 : -1), (googlyEyes ? 14 : -1));
    }

    __global__ void renderPoseBody25(float* targetPtr, const int targetWidth, const int targetHeight,
                                     const float* const posePtr, const int numberPeople, const float threshold,
                                     const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(BODY_25_PAIRS_GPU) / (2*sizeof(BODY_25_PAIRS_GPU[0]));
        const auto numberScales = sizeof(BODY_25_SCALES) / sizeof(BODY_25_SCALES[0]);
        const auto numberColors = sizeof(BODY_25_COLORS) / (3*sizeof(BODY_25_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, BODY_25_PAIRS_GPU, numberPeople, 25, numberPartPairs, BODY_25_COLORS, numberColors,
                        radius, lineWidth, BODY_25_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 15 : -1), (googlyEyes ? 16 : -1));
    }

    __global__ void renderPoseBody59(float* targetPtr, const int targetWidth, const int targetHeight,
                                     const float* const posePtr, const int numberPeople, const float threshold,
                                     const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(BODY_59_PAIRS_GPU) / (2*sizeof(BODY_59_PAIRS_GPU[0]));
        const auto numberScales = sizeof(BODY_59_SCALES) / sizeof(BODY_59_SCALES[0]);
        const auto numberColors = sizeof(BODY_59_COLORS) / (3*sizeof(BODY_59_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, BODY_59_PAIRS_GPU, numberPeople, 59, numberPartPairs, BODY_59_COLORS, numberColors,
                        radius, lineWidth, BODY_59_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 15 : -1), (googlyEyes ? 16 : -1));
    }

    __global__ void renderPoseBody65(float* targetPtr, const int targetWidth, const int targetHeight,
                                     const float* const posePtr, const int numberPeople, const float threshold,
                                     const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(BODY_65_PAIRS_GPU) / (2*sizeof(BODY_65_PAIRS_GPU[0]));
        const auto numberScales = sizeof(BODY_65_SCALES) / sizeof(BODY_65_SCALES[0]);
        const auto numberColors = sizeof(BODY_65_COLORS) / (3*sizeof(BODY_65_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, BODY_65_PAIRS_GPU, numberPeople, 65, numberPartPairs, BODY_65_COLORS, numberColors,
                        radius, lineWidth, BODY_65_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 15 : -1), (googlyEyes ? 16 : -1));
    }

    __global__ void renderPoseMpi29Parts(float* targetPtr, const int targetWidth, const int targetHeight,
                                         const float* const posePtr, const int numberPeople, const float threshold,
                                         const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(MPI_PAIRS_GPU) / (2*sizeof(MPI_PAIRS_GPU[0]));
        const auto numberScales = sizeof(MPI_SCALES) / sizeof(MPI_SCALES[0]);
        const auto numberColors = sizeof(MPI_COLORS) / (3*sizeof(MPI_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, MPI_PAIRS_GPU, numberPeople, 15, numberPartPairs, MPI_COLORS, numberColors,
                        radius, lineWidth, COCO_SCALES, numberScales, threshold, alphaColorToAdd, blendOriginalFrame);
    }

    __global__ void renderPoseCar12(float* targetPtr, const int targetWidth, const int targetHeight,
                                    const float* const posePtr, const int numberPeople, const float threshold,
                                    const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(CAR_12_PAIRS_GPU) / (2*sizeof(CAR_12_PAIRS_GPU[0]));
        const auto numberScales = sizeof(CAR_12_SCALES) / sizeof(CAR_12_SCALES[0]);
        const auto numberColors = sizeof(CAR_12_COLORS) / (3*sizeof(CAR_12_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, CAR_12_PAIRS_GPU, numberPeople, 12, numberPartPairs, CAR_12_COLORS, numberColors,
                        radius, lineWidth, CAR_12_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 4 : -1), (googlyEyes ? 5 : -1));
    }

    __global__ void renderPoseCar22(float* targetPtr, const int targetWidth, const int targetHeight,
                                    const float* const posePtr, const int numberPeople, const float threshold,
                                    const bool googlyEyes, const bool blendOriginalFrame, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto globalIdx = threadIdx.y * blockDim.x + threadIdx.x;

        // Shared parameters
        __shared__ float2 sharedMins[POSE_MAX_PEOPLE];
        __shared__ float2 sharedMaxs[POSE_MAX_PEOPLE];
        __shared__ float sharedScaleF[POSE_MAX_PEOPLE];

        // Other parameters
        const auto numberPartPairs = sizeof(CAR_22_PAIRS_GPU) / (2*sizeof(CAR_22_PAIRS_GPU[0]));
        const auto numberScales = sizeof(CAR_22_SCALES) / sizeof(CAR_22_SCALES[0]);
        const auto numberColors = sizeof(CAR_22_COLORS) / (3*sizeof(CAR_22_COLORS[0]));
        const auto radius = fastMin(targetWidth, targetHeight) / 100.f;
        const auto lineWidth = fastMin(targetWidth, targetHeight) / 120.f;

        // Render key points
        renderKeypoints(targetPtr, sharedMaxs, sharedMins, sharedScaleF, globalIdx, x, y, targetWidth, targetHeight,
                        posePtr, CAR_22_PAIRS_GPU, numberPeople, 22, numberPartPairs, CAR_22_COLORS, numberColors,
                        radius, lineWidth, CAR_22_SCALES, numberScales, threshold, alphaColorToAdd,
                        blendOriginalFrame, (googlyEyes ? 6 : -1), (googlyEyes ? 7 : -1));
    }

    __global__ void renderBodyPartHeatMaps(float* targetPtr, const int targetWidth, const int targetHeight,
                                           const float* const heatMapPtr, const int widthHeatMap,
                                           const int heightHeatMap, const float scaleToKeepRatio,
                                           const int numberBodyParts, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        const auto numberColors = sizeof(COCO_COLORS)/(3*sizeof(COCO_COLORS[0]));

        if (x < targetWidth && y < targetHeight)
        {
            float rgbColor [3] = {0.f,0.f,0.f};
            const auto xSource = (x + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto ySource = (y + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto xHeatMap = fastTruncate(int(xSource + 1e-5), 0, widthHeatMap);
            const auto yHeatMap = fastTruncate(int(ySource + 1e-5), 0, heightHeatMap);
            const auto heatMapArea = widthHeatMap * heightHeatMap;
            for (auto part = 0u ; part < numberBodyParts ; part++)
            {
                const auto offsetOrigin = part * heatMapArea;
                // __saturatef = trucate to [0,1]
                const auto value = __saturatef(heatMapPtr[offsetOrigin + yHeatMap*widthHeatMap + xHeatMap]);
                const auto rgbColorIndex = (part%numberColors)*3;
                rgbColor[0] += value*COCO_COLORS[rgbColorIndex];
                rgbColor[1] += value*COCO_COLORS[rgbColorIndex+1];
                rgbColor[2] += value*COCO_COLORS[rgbColorIndex+2];
            }

            const auto blueIndex = 3*(y * targetWidth + x);
            addColorWeighted(targetPtr[blueIndex+2], targetPtr[blueIndex+1], targetPtr[blueIndex], rgbColor,
                             alphaColorToAdd);
        }
    }

    __global__ void renderBodyPartHeatMap(float* targetPtr, const int targetWidth, const int targetHeight,
                                          const float* const heatMapPtr, const int widthHeatMap,
                                          const int heightHeatMap, const float scaleToKeepRatio, const unsigned int part,
                                          const float alphaColorToAdd, const bool absValue = false)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const auto xSource = (x + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto ySource = (y + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto* const heatMapPtrOffsetted = heatMapPtr + part * widthHeatMap * heightHeatMap;
            const auto interpolatedValue = bicubicInterpolate(heatMapPtrOffsetted, xSource, ySource, widthHeatMap,
                                                              heightHeatMap, widthHeatMap);

            float rgbColor[3];
            if (absValue)
                getColorHeatMap(rgbColor, fabsf(interpolatedValue), 0.f, 1.f);
            else
                getColorHeatMap(rgbColor, interpolatedValue, 0.f, 1.f);

            const auto blueIndex = 3*(y * targetWidth + x);
            addColorWeighted(targetPtr[blueIndex+2], targetPtr[blueIndex+1], targetPtr[blueIndex], rgbColor,
                             alphaColorToAdd);
        }
    }

    __global__ void renderPartAffinities(float* targetPtr, const int targetWidth, const int targetHeight,
                                         const float* const heatMapPtr, const int widthHeatMap,
                                         const int heightHeatMap, const float scaleToKeepRatio,
                                         const int partsToRender, const int initPart, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            float rgbColor [3] = {0.f,0.f,0.f};
            const auto xSource = (x + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto ySource = (y + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto heatMapArea = widthHeatMap * heightHeatMap;

            for (auto part = initPart ; part < initPart + partsToRender*2 ; part += 2)
            {
                int xIntArray[4];
                int yIntArray[4];
                float dx;
                float dy;
                cubicSequentialData(xIntArray, yIntArray, dx, dy, xSource, ySource, widthHeatMap, heightHeatMap);

                const auto offsetOriginX = part * heatMapArea;
                const auto offsetOriginY = (part+1) * heatMapArea;
                auto valueX = heatMapPtr[offsetOriginX + yIntArray[1]*widthHeatMap + xIntArray[1]];
                auto valueY = heatMapPtr[offsetOriginY + yIntArray[1]*widthHeatMap + xIntArray[1]];
                if (partsToRender == 1)
                {
                    const auto xB = heatMapPtr[offsetOriginX + yIntArray[1]*widthHeatMap + xIntArray[2]];
                    const auto xC = heatMapPtr[offsetOriginX + yIntArray[2]*widthHeatMap + xIntArray[1]];
                    const auto xD = heatMapPtr[offsetOriginX + yIntArray[2]*widthHeatMap + xIntArray[2]];
                    valueX = (1-dx)*(1-dy)*valueX
                           + dx*(1-dy)*xB
                           + (1-dx)*dy*xC
                           + dx*dy*xD;
                    const auto yB = heatMapPtr[offsetOriginY + yIntArray[1]*widthHeatMap + xIntArray[2]];
                    const auto yC = heatMapPtr[offsetOriginY + yIntArray[2]*widthHeatMap + xIntArray[1]];
                    const auto yD = heatMapPtr[offsetOriginY + yIntArray[2]*widthHeatMap + xIntArray[2]];
                    valueY = (1-dx)*(1-dy)*valueY
                           + dx*(1-dy)*yB
                           + (1-dx)*dy*yC
                           + dx*dy*yD;
                }

                float3 rgbColor2;
                // if (forceNorm1)
                // {
                //     const auto norm = std::sqrt(valueX*valueX + valueY*valueY);
                //     if (norm > 0.05f)
                //         getColorXYAffinity(rgbColor2, valueX/norm, valueY/norm);
                //     else
                //         getColorXYAffinity(rgbColor2, valueX, valueY);
                // }
                // else
                getColorXYAffinity(rgbColor2, valueX, valueY);
                rgbColor[0] += rgbColor2.x;
                rgbColor[1] += rgbColor2.y;
                rgbColor[2] += rgbColor2.z;
            }

            const auto blueIndex = 3*(y * targetWidth + x);
            addColorWeighted(targetPtr[blueIndex+2], targetPtr[blueIndex+1], targetPtr[blueIndex], rgbColor,
                             alphaColorToAdd);
        }
    }

    __global__ void renderDistance(float* targetPtr, const int targetWidth, const int targetHeight,
                                   const float* const heatMapPtr, const int widthHeatMap, const int heightHeatMap,
                                   const float scaleToKeepRatio, const int part, const int numberBodyParts,
                                   const int numberBodyPAFChannels, const float alphaColorToAdd)
    {
        const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if (x < targetWidth && y < targetHeight)
        {
            const auto xSource = (x + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto ySource = (y + 0.5f) / scaleToKeepRatio - 0.5f;
            const auto heatMapOffset = part * widthHeatMap * heightHeatMap
                                     + (numberBodyParts+1+numberBodyPAFChannels)*widthHeatMap * heightHeatMap;
            const auto* const heatMapPtrOffsetted = heatMapPtr + heatMapOffset;
            const auto interpolatedValue = 0.5f
                                         + 0.5f * bicubicInterpolate(heatMapPtrOffsetted, xSource, ySource,
                                                                     widthHeatMap, heightHeatMap, widthHeatMap);

            float rgbColor[3];
            getColorHeatMap(rgbColor, interpolatedValue, 0.f, 1.f);

            const auto blueIndex = 3*(y * targetWidth + x);
            addColorWeighted(targetPtr[blueIndex+2], targetPtr[blueIndex+1], targetPtr[blueIndex], rgbColor,
                             alphaColorToAdd);
        }
    }

    inline void checkAlpha(const float alphaColorToAdd)
    {
        if (alphaColorToAdd < 0.f || alphaColorToAdd > 1.f)
            error("Alpha must be in the range [0, 1].", __LINE__, __FUNCTION__, __FILE__);
    }

    inline void renderPosePAFGpuAux(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize,
                                    const float* const heatMapPtr, const Point<int>& heatMapSize,
                                    const float scaleToKeepRatio, const int part, const int partsToRender,
                                    const float alphaBlending)
    {
        try
        {
            //framePtr      =   width * height * 3
            //heatMapPtr    =   heatMapSize.x * heatMapSize.y * #body parts
            checkAlpha(alphaBlending);
            const auto heatMapOffset = getPoseNumberBodyParts(poseModel) * heatMapSize.area();
            dim3 threadsPerBlock;
            dim3 numBlocks;
            getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);
            renderPartAffinities<<<threadsPerBlock, numBlocks>>>(framePtr, frameSize.x, frameSize.y, heatMapPtr,
                                                                 heatMapSize.x, heatMapSize.y, scaleToKeepRatio,
                                                                 partsToRender, part, alphaBlending);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderPoseKeypointsGpu(float* framePtr, const PoseModel poseModel, const int numberPeople,
                                const Point<int>& frameSize, const float* const posePtr, const float renderThreshold,
                                const bool googlyEyes, const bool blendOriginalFrame, const float alphaBlending)
    {
        try
        {
            if (numberPeople > 0 || !blendOriginalFrame)
            {
                // framePtr      =   width * height * 3
                // heatMapPtr    =   heatMapSize.x * heatMapSize.y * #body parts
                // posePtr       =   3 (x,y,score) * #Body parts * numberPeople
                if (googlyEyes && (poseModel == PoseModel::MPI_15 || poseModel == PoseModel::MPI_15_4))
                    error("Bool googlyEyes not compatible with MPI models.",
                          __LINE__, __FUNCTION__, __FILE__);
                if (numberPeople > POSE_MAX_PEOPLE)
                    error("Rendering assumes that numberPeople <= POSE_MAX_PEOPLE = " + std::to_string(POSE_MAX_PEOPLE)
                          + ".", __LINE__, __FUNCTION__, __FILE__);

                dim3 threadsPerBlock;
                dim3 numBlocks;
                getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);

                // Body pose
                if (poseModel == PoseModel::BODY_25 || poseModel == PoseModel::BODY_25_19
                    || poseModel == PoseModel::BODY_25D || poseModel == PoseModel::BODY_25E)
                    renderPoseBody25<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::COCO_18)
                    renderPoseCoco<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::BODY_19 || poseModel == PoseModel::BODY_19_X2
                         || poseModel == PoseModel::BODY_19N)
                    renderPoseBody19<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::BODY_23)
                    renderPoseBody23<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::BODY_59)
                    renderPoseBody59<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::BODY_65)
                    renderPoseBody65<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::MPI_15 || poseModel == PoseModel::MPI_15_4)
                    renderPoseMpi29Parts<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold,
                        blendOriginalFrame, alphaBlending
                    );
                // Car pose
                else if (poseModel == PoseModel::CAR_12)
                    renderPoseCar12<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                else if (poseModel == PoseModel::CAR_22)
                    renderPoseCar22<<<threadsPerBlock, numBlocks>>>(
                        framePtr, frameSize.x, frameSize.y, posePtr, numberPeople, renderThreshold, googlyEyes,
                        blendOriginalFrame, alphaBlending
                    );
                // Unknown
                else
                    error("Invalid Model.", __LINE__, __FUNCTION__, __FILE__);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderPoseHeatMapGpu(float* framePtr, const Point<int>& frameSize, const float* const heatMapPtr,
                              const Point<int>& heatMapSize, const float scaleToKeepRatio, const unsigned int part,
                              const float alphaBlending)
    {
        try
        {
            //framePtr      =   width * height * 3
            //heatMapPtr    =   heatMapSize.x * heatMapSize.y * #body parts
            checkAlpha(alphaBlending);
            dim3 threadsPerBlock;
            dim3 numBlocks;
            getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);

            renderBodyPartHeatMap<<<threadsPerBlock, numBlocks>>>(
                framePtr, frameSize.x, frameSize.y, heatMapPtr, heatMapSize.x, heatMapSize.y, scaleToKeepRatio,
                part, alphaBlending
            );
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderPoseHeatMapsGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize,
                               const float* const heatMapPtr, const Point<int>& heatMapSize,
                               const float scaleToKeepRatio, const float alphaBlending)
    {
        try
        {
            //framePtr      =   width * height * 3
            //heatMapPtr    =   heatMapSize.x * heatMapSize.y * #body parts
            checkAlpha(alphaBlending);
            dim3 threadsPerBlock;
            dim3 numBlocks;
            getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto heatMapOffset = numberBodyParts * heatMapSize.area();

            renderBodyPartHeatMaps<<<threadsPerBlock, numBlocks>>>(
                framePtr, frameSize.x, frameSize.y, heatMapPtr, heatMapSize.x, heatMapSize.y, scaleToKeepRatio,
                numberBodyParts, alphaBlending
            );
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderPosePAFGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize,
                          const float* const heatMapPtr, const Point<int>& heatMapSize, const float scaleToKeepRatio,
                          const int part, const float alphaBlending)
    {
        try
        {
            renderPosePAFGpuAux(framePtr, poseModel, frameSize, heatMapPtr, heatMapSize, scaleToKeepRatio, part, 1,
                                alphaBlending);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderPosePAFsGpu(float* framePtr, const PoseModel poseModel, const Point<int>& frameSize,
                           const float* const heatMapPtr, const Point<int>& heatMapSize, const float scaleToKeepRatio,
                           const float alphaBlending)
    {
        try
        {
            const auto numberBodyPartPairs = (int)getPosePartPairs(poseModel).size()/2;
            renderPosePAFGpuAux(framePtr, poseModel, frameSize, heatMapPtr, heatMapSize, scaleToKeepRatio,
                                getPoseNumberBodyParts(poseModel)+1, numberBodyPartPairs, alphaBlending);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void renderPoseDistanceGpu(float* framePtr, const Point<int>& frameSize, const float* const heatMapPtr,
                               const Point<int>& heatMapSize, const float scaleToKeepRatio, const unsigned int part,
                               const float alphaBlending)
    {
        try
        {
            // // As PAF
            // const bool forceNorm1 = true;
            // renderPosePAFGpuAux(framePtr, poseModel, frameSize, heatMapPtr, heatMapSize, scaleToKeepRatio, part, 1,
            //                     alphaBlending, forceNorm1);

            // As body part
            // framePtr      =   width * height * 3
            // heatMapPtr    =   heatMapSize.x * heatMapSize.y * #body parts
            checkAlpha(alphaBlending);
            dim3 threadsPerBlock;
            dim3 numBlocks;
            getNumberCudaThreadsAndBlocks(threadsPerBlock, numBlocks, frameSize);

            const auto absValue = true;
            renderBodyPartHeatMap<<<threadsPerBlock, numBlocks>>>(
                framePtr, frameSize.x, frameSize.y, heatMapPtr, heatMapSize.x, heatMapSize.y, scaleToKeepRatio,
                part, alphaBlending, absValue);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
