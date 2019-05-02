#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>

namespace op
{
    template<typename T>
    inline __device__ int intRoundGPU(const T a)
    {
        return int(a+T(0.5));
    }

    template <typename T>
    inline __device__  T process(
        const T* bodyPartA, const T* bodyPartB, const T* mapX, const T* mapY, const int heatmapWidth,
        const int heatmapHeight, const T interThreshold, const T interMinAboveThreshold)
    {
        const auto vectorAToBX = bodyPartB[0] - bodyPartA[0];
        const auto vectorAToBY = bodyPartB[1] - bodyPartA[1];
        const auto vectorAToBMax = max(abs(vectorAToBX), abs(vectorAToBY));
        const auto numberPointsInLine = max(5, min(25, intRoundGPU(sqrt(5*vectorAToBMax))));
        const auto vectorNorm = T(sqrt(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY));

        if (vectorNorm > 1e-6)
        {
            const auto sX = bodyPartA[0];
            const auto sY = bodyPartA[1];
            const auto vectorAToBNormX = vectorAToBX/vectorNorm;
            const auto vectorAToBNormY = vectorAToBY/vectorNorm;

            auto sum = T(0.);
            auto count = 0;
            const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
            const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
            for (auto lm = 0; lm < numberPointsInLine; lm++)
            {
                const auto mX = min(heatmapWidth-1, intRoundGPU(sX + lm*vectorAToBXInLine));
                const auto mY = min(heatmapHeight-1, intRoundGPU(sY + lm*vectorAToBYInLine));
                const auto idx = mY * heatmapWidth + mX;
                const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                if (score > interThreshold)
                {
                    sum += score;
                    count++;
                }
            }

            // Return PAF score
            if (count/T(numberPointsInLine) > interMinAboveThreshold)
                return sum/count;
            else
            {
                // Ideally, if distanceAB = 0, PAF is 0 between A and B, provoking a false negative
                // To fix it, we consider PAF-connected keypoints very close to have a minimum PAF score, such that:
                //     1. It will consider very close keypoints (where the PAF is 0)
                //     2. But it will not automatically connect them (case PAF score = 1), or real PAF might got
                //        missing
                const auto l2Dist = sqrtf(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY);
                const auto threshold = sqrtf(heatmapWidth*heatmapHeight)/150; // 3.3 for 368x656, 6.6 for 2x resolution
                if (l2Dist < threshold)
                    return T(0.15);
            }
        }
        return -1;
    }

    // template <typename T>
    // __global__ void pafScoreKernelOld(
    //     T* pairScoresPtr, const T* const heatMapPtr, const T* const peaksPtr, const unsigned int* const bodyPartPairsPtr,
    //     const unsigned int* const mapIdxPtr, const unsigned int maxPeaks, const int numberBodyPartPairs,
    //     const int heatmapWidth, const int heatmapHeight, const T interThreshold, const T interMinAboveThreshold)
    // {
    //     const auto pairIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    //     const auto peakA = (blockIdx.y * blockDim.y) + threadIdx.y;
    //     const auto peakB = (blockIdx.z * blockDim.z) + threadIdx.z;

    //     if (pairIndex < numberBodyPartPairs && peakA < maxPeaks && peakB < maxPeaks)
    //     {
    //         const auto baseIndex = 2*pairIndex;
    //         const auto partA = bodyPartPairsPtr[baseIndex];
    //         const auto partB = bodyPartPairsPtr[baseIndex + 1];

    //         const T numberPeaksA = peaksPtr[3*partA*(maxPeaks+1)];
    //         const T numberPeaksB = peaksPtr[3*partB*(maxPeaks+1)];

    //         const auto outputIndex = (pairIndex*maxPeaks+peakA)*maxPeaks + peakB;
    //         if (peakA < numberPeaksA && peakB < numberPeaksB)
    //         {
    //             const auto mapIdxX = mapIdxPtr[baseIndex];
    //             const auto mapIdxY = mapIdxPtr[baseIndex + 1];

    //             const T* const bodyPartA = peaksPtr + (3*(partA*(maxPeaks+1) + peakA+1));
    //             const T* const bodyPartB = peaksPtr + (3*(partB*(maxPeaks+1) + peakB+1));
    //             const T* const mapX = heatMapPtr + mapIdxX*heatmapWidth*heatmapHeight;
    //             const T* const mapY = heatMapPtr + mapIdxY*heatmapWidth*heatmapHeight;
    //             pairScoresPtr[outputIndex] = process(
    //                 bodyPartA, bodyPartB, mapX, mapY, heatmapWidth, heatmapHeight, interThreshold,
    //                 interMinAboveThreshold);
    //         }
    //         else
    //             pairScoresPtr[outputIndex] = -1;
    //     }
    // }

    template <typename T>
    __global__ void pafScoreKernel(
        T* pairScoresPtr, const T* const heatMapPtr, const T* const peaksPtr, const unsigned int* const bodyPartPairsPtr,
        const unsigned int* const mapIdxPtr, const unsigned int maxPeaks, const int numberBodyPartPairs,
        const int heatmapWidth, const int heatmapHeight, const T interThreshold, const T interMinAboveThreshold)
    {
        const auto peakB = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto peakA = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto pairIndex = (blockIdx.z * blockDim.z) + threadIdx.z;

        if (peakA < maxPeaks && peakB < maxPeaks)
        // if (pairIndex < numberBodyPartPairs && peakA < maxPeaks && peakB < maxPeaks)
        {
            const auto baseIndex = 2*pairIndex;
            const auto partA = bodyPartPairsPtr[baseIndex];
            const auto partB = bodyPartPairsPtr[baseIndex + 1];

            const T numberPeaksA = peaksPtr[3*partA*(maxPeaks+1)];
            const T numberPeaksB = peaksPtr[3*partB*(maxPeaks+1)];

            const auto outputIndex = (pairIndex*maxPeaks+peakA)*maxPeaks + peakB;
            if (peakA < numberPeaksA && peakB < numberPeaksB)
            {
                const auto mapIdxX = mapIdxPtr[baseIndex];
                const auto mapIdxY = mapIdxPtr[baseIndex + 1];

                const T* const bodyPartA = peaksPtr + (3*(partA*(maxPeaks+1) + peakA+1));
                const T* const bodyPartB = peaksPtr + (3*(partB*(maxPeaks+1) + peakB+1));
                const T* const mapX = heatMapPtr + mapIdxX*heatmapWidth*heatmapHeight;
                const T* const mapY = heatMapPtr + mapIdxY*heatmapWidth*heatmapHeight;
                pairScoresPtr[outputIndex] = process(
                    bodyPartA, bodyPartB, mapX, mapY, heatmapWidth, heatmapHeight, interThreshold,
                    interMinAboveThreshold);
            }
            else
                pairScoresPtr[outputIndex] = -1;
        }
    }

    // template <typename T>
    // std::vector<std::pair<std::vector<int>, T>> pafVectorIntoPeopleVectorOld(
    //     const std::vector<std::tuple<T, T, int, int, int>>& pairConnections, const T* const peaksPtr,
    //     const int maxPeaks, const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts)
    // {
    //     try
    //     {
    //         // std::vector<std::pair<std::vector<int>, double>> refers to:
    //         //     - std::vector<int>: [body parts locations, #body parts found]
    //         //     - double: person subset score
    //         std::vector<std::pair<std::vector<int>, T>> peopleVector;
    //         const auto vectorSize = numberBodyParts+1;
    //         const auto peaksOffset = (maxPeaks+1);
    //         // Save which body parts have been already assigned
    //         std::vector<int> personAssigned(numberBodyParts*maxPeaks, -1);
    //         // Iterate over each PAF pair connection detected
    //         // E.g., neck1-nose2, neck5-Lshoulder0, etc.
    //         for (const auto& pairConnection : pairConnections)
    //         {
    //             // Read pairConnection
    //             // // Total score - only required for previous sort
    //             // const auto totalScore = std::get<0>(pairConnection);
    //             const auto pafScore = std::get<1>(pairConnection);
    //             const auto pairIndex = std::get<2>(pairConnection);
    //             const auto indexA = std::get<3>(pairConnection);
    //             const auto indexB = std::get<4>(pairConnection);
    //             // Derived data
    //             const auto bodyPartA = bodyPartPairs[2*pairIndex];
    //             const auto bodyPartB = bodyPartPairs[2*pairIndex+1];

    //             const auto indexScoreA = (bodyPartA*peaksOffset + indexA)*3 + 2;
    //             const auto indexScoreB = (bodyPartB*peaksOffset + indexB)*3 + 2;
    //             // -1 because indexA and indexB are 1-based
    //             auto& aAssigned = personAssigned[bodyPartA*maxPeaks+indexA-1];
    //             auto& bAssigned = personAssigned[bodyPartB*maxPeaks+indexB-1];
    //             // Debugging
    //             #ifdef DEBUG
    //                 if (indexA-1 > peaksOffset || indexA <= 0)
    //                     error("Something is wrong: " + std::to_string(indexA)
    //                           + " vs. " + std::to_string(peaksOffset) + ". Contact us.",
    //                           __LINE__, __FUNCTION__, __FILE__);
    //                 if (indexB-1 > peaksOffset || indexB <= 0)
    //                     error("Something is wrong: " + std::to_string(indexB)
    //                           + " vs. " + std::to_string(peaksOffset) + ". Contact us.",
    //                           __LINE__, __FUNCTION__, __FILE__);
    //             #endif

    //             // Different cases:
    //             //     1. A & B not assigned yet: Create new person
    //             //     2. A assigned but not B: Add B to person with A (if no another B there)
    //             //     3. B assigned but not A: Add A to person with B (if no another A there)
    //             //     4. A & B already assigned to same person (circular/redundant PAF): Update person score
    //             //     5. A & B already assigned to different people: Merge people if keypoint intersection is null
    //             // 1. A & B not assigned yet: Create new person
    //             if (aAssigned < 0 && bAssigned < 0)
    //             {
    //                 // Keypoint indexes
    //                 std::vector<int> rowVector(vectorSize, 0);
    //                 rowVector[bodyPartA] = indexScoreA;
    //                 rowVector[bodyPartB] = indexScoreB;
    //                 // Number keypoints
    //                 rowVector.back() = 2;
    //                 // Score
    //                 const auto personScore = peaksPtr[indexScoreA] + peaksPtr[indexScoreB] + pafScore;
    //                 // Set associated personAssigned as assigned
    //                 aAssigned = (int)peopleVector.size();
    //                 bAssigned = aAssigned;
    //                 // Create new personVector
    //                 peopleVector.emplace_back(std::make_pair(rowVector, personScore));
    //             }
    //             // 2. A assigned but not B: Add B to person with A (if no another B there)
    //             // or
    //             // 3. B assigned but not A: Add A to person with B (if no another A there)
    //             else if ((aAssigned >= 0 && bAssigned < 0)
    //                 || (aAssigned < 0 && bAssigned >= 0))
    //             {
    //                 // Assign person1 to one where xAssigned >= 0
    //                 const auto assigned1 = (aAssigned >= 0 ? aAssigned : bAssigned);
    //                 auto& assigned2 = (aAssigned >= 0 ? bAssigned : aAssigned);
    //                 const auto bodyPart2 = (aAssigned >= 0 ? bodyPartB : bodyPartA);
    //                 const auto indexScore2 = (aAssigned >= 0 ? indexScoreB : indexScoreA);
    //                 // Person index
    //                 auto& personVector = peopleVector[assigned1];
    //                 // Debugging
    //                 #ifdef DEBUG
    //                     const auto bodyPart1 = (aAssigned >= 0 ? bodyPartA : bodyPartB);
    //                     const auto indexScore1 = (aAssigned >= 0 ? indexScoreA : indexScoreB);
    //                     const auto index1 = (aAssigned >= 0 ? indexA : indexB);
    //                     if ((unsigned int)personVector.first.at(bodyPart1) != indexScore1)
    //                         error("Something is wrong: "
    //                               + std::to_string((personVector.first[bodyPart1]-2)/3-bodyPart1*peaksOffset)
    //                               + " vs. " + std::to_string((indexScore1-2)/3-bodyPart1*peaksOffset) + " vs. "
    //                               + std::to_string(index1) + ". Contact us.",
    //                               __LINE__, __FUNCTION__, __FILE__);
    //                 #endif
    //                 // If person with 1 does not have a 2 yet
    //                 if (personVector.first[bodyPart2] == 0)
    //                 {
    //                     // Update keypoint indexes
    //                     personVector.first[bodyPart2] = indexScore2;
    //                     // Update number keypoints
    //                     personVector.first.back()++;
    //                     // Update score
    //                     personVector.second += peaksPtr[indexScore2] + pafScore;
    //                     // Set associated personAssigned as assigned
    //                     assigned2 = assigned1;
    //                 }
    //                 // Otherwise, ignore this B because the previous one came from a higher PAF-confident score
    //             }
    //             // 4. A & B already assigned to same person (circular/redundant PAF): Update person score
    //             else if (aAssigned >=0 && bAssigned >=0 && aAssigned == bAssigned)
    //                 peopleVector[aAssigned].second += pafScore;
    //             // 5. A & B already assigned to different people: Merge people if keypoint intersection is null
    //             // I.e., that the keypoints in person A and B do not overlap
    //             else if (aAssigned >=0 && bAssigned >=0 && aAssigned != bAssigned)
    //             {
    //                 // Assign person1 to the one with lowest index for 2 reasons:
    //                 //     1. Speed up: Removing an element from std::vector is cheaper for latest elements
    //                 //     2. Avoid harder index update: Updated elements in person1ssigned would depend on
    //                 //        whether person1 > person2 or not: element = aAssigned - (person2 > person1 ? 1 : 0)
    //                 const auto assigned1 = (aAssigned < bAssigned ? aAssigned : bAssigned);
    //                 const auto assigned2 = (aAssigned < bAssigned ? bAssigned : aAssigned);
    //                 auto& person1 = peopleVector[assigned1].first;
    //                 const auto& person2 = peopleVector[assigned2].first;
    //                 // Check if complementary
    //                 // Defining found keypoint indexes in personA as kA, and analogously kB
    //                 // Complementary if and only if kA intersection kB = empty. I.e., no common keypoints
    //                 bool complementary = true;
    //                 for (auto part = 0u ; part < numberBodyParts ; part++)
    //                 {
    //                     if (person1[part] > 0 && person2[part] > 0)
    //                     {
    //                         complementary = false;
    //                         break;
    //                     }
    //                 }
    //                 // If complementary, merge both people into 1
    //                 if (complementary)
    //                 {
    //                     // Update keypoint indexes
    //                     for (auto part = 0u ; part < numberBodyParts ; part++)
    //                         if (person1[part] == 0)
    //                             person1[part] = person2[part];
    //                     // Update number keypoints
    //                     person1.back() += person2.back();
    //                     // Update score
    //                     peopleVector[assigned1].second += peopleVector[assigned2].second + pafScore;
    //                     // Erase the non-merged person
    //                     peopleVector.erase(peopleVector.begin()+assigned2);
    //                     // Update associated personAssigned (person indexes have changed)
    //                     for (auto& element : personAssigned)
    //                     {
    //                         if (element == assigned2)
    //                             element = assigned1;
    //                         else if (element > assigned2)
    //                             element--;
    //                     }
    //                 }
    //             }
    //         }
    //         // Return result
    //         return peopleVector;
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //         return {};
    //     }
    // }

    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapGpuPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const bool maximizePositives, Array<T> pairScoresCpu, T* pairScoresGpuPtr,
                             const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr,
                             const T* const peaksGpuPtr)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = (unsigned int)(bodyPartPairs.size() / 2);
            const auto totalComputations = pairScoresCpu.getVolume();

            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);
            if (bodyPartPairsGpuPtr == nullptr || mapIdxGpuPtr == nullptr)
                error("The pointers bodyPartPairsGpuPtr and mapIdxGpuPtr cannot be nullptr.",
                      __LINE__, __FUNCTION__, __FILE__);

            // const auto REPS = 1000;
            // double timeNormalize0 = 0.;
            // double timeNormalize1 = 0.;
            // double timeNormalize2 = 0.;

            // // Old - Non-efficient code
            // OP_CUDA_PROFILE_INIT(REPS);
            // // Run Kernel - pairScoresGpu
            // const dim3 THREADS_PER_BLOCK{4, 16, 16};
            // const dim3 numBlocks{
            //     getNumberCudaBlocks(numberBodyPartPairs, THREADS_PER_BLOCK.x),
            //     getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.y),
            //     getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.z)};
            // pafScoreKernelOld<<<numBlocks, THREADS_PER_BLOCK>>>(
            //     pairScoresGpuPtr, heatMapGpuPtr, peaksGpuPtr, bodyPartPairsGpuPtr, mapIdxGpuPtr,
            //     maxPeaks, (int)numberBodyPartPairs, heatMapSize.x, heatMapSize.y, interThreshold,
            //     interMinAboveThreshold);
            // // pairScoresCpu <-- pairScoresGpu
            // cudaMemcpy(pairScoresCpu.getPtr(), pairScoresGpuPtr, totalComputations * sizeof(T),
            //            cudaMemcpyDeviceToHost);
            // // Get pair connections and their scores
            // const auto pairConnections = pafPtrIntoVector(
            //     pairScoresCpu, peaksPtr, maxPeaks, bodyPartPairs, numberBodyPartPairs);
            // const auto peopleVector = pafVectorIntoPeopleVectorOld(
            //     pairConnections, peaksPtr, maxPeaks, bodyPartPairs, numberBodyParts);
            // // Delete people below the following thresholds:
            //     // a) minSubsetCnt: removed if less than minSubsetCnt body parts
            //     // b) minSubsetScore: removed if global score smaller than this
            //     // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            // int numberPeople;
            // std::vector<int> validSubsetIndexes;
            // validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, peopleVector.size()));
            // removePeopleBelowThresholds(validSubsetIndexes, numberPeople, peopleVector, numberBodyParts, minSubsetCnt,
            //                             minSubsetScore, maxPeaks, maximizePositives);
            // // Fill and return poseKeypoints
            // peopleVectorToPeopleArray(poseKeypoints, poseScores, scaleFactor, peopleVector, validSubsetIndexes,
            //                           peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);
            // OP_PROFILE_END(timeNormalize1, 1e3, REPS);

            // Efficient code
            // OP_CUDA_PROFILE_INIT(REPS);
            // Run Kernel - pairScoresGpu
            const dim3 THREADS_PER_BLOCK{128, 1, 1};
            const dim3 numBlocks{
                getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.x),
                getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.y),
                getNumberCudaBlocks(numberBodyPartPairs, THREADS_PER_BLOCK.z)};
            pafScoreKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
                pairScoresGpuPtr, heatMapGpuPtr, peaksGpuPtr, bodyPartPairsGpuPtr, mapIdxGpuPtr,
                maxPeaks, (int)numberBodyPartPairs, heatMapSize.x, heatMapSize.y, interThreshold,
                interMinAboveThreshold);
            // pairScoresCpu <-- pairScoresGpu
            cudaMemcpy(pairScoresCpu.getPtr(), pairScoresGpuPtr, totalComputations * sizeof(T),
                       cudaMemcpyDeviceToHost);
            // Get pair connections and their scores
            const auto pairConnections = pafPtrIntoVector(
                pairScoresCpu, peaksPtr, maxPeaks, bodyPartPairs, numberBodyPartPairs);
            const auto peopleVector = pafVectorIntoPeopleVector(
                pairConnections, peaksPtr, maxPeaks, bodyPartPairs, numberBodyParts);
            // // Old code: Get pair connections and their scores
            // // std::vector<std::pair<std::vector<int>, double>> refers to:
            // //     - std::vector<int>: [body parts locations, #body parts found]
            // //     - double: person subset score
            // const T* const tNullptr = nullptr;
            // const auto peopleVector = createPeopleVector(
            //     tNullptr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
            //     bodyPartPairs, numberBodyParts, numberBodyPartPairs, pairScoresCpu);
            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, peopleVector.size()));
            removePeopleBelowThresholds(validSubsetIndexes, numberPeople, peopleVector, numberBodyParts, minSubsetCnt,
                                        minSubsetScore, maxPeaks, maximizePositives);
            // Fill and return poseKeypoints
            peopleVectorToPeopleArray(poseKeypoints, poseScores, scaleFactor, peopleVector, validSubsetIndexes,
                                      peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);
            // OP_PROFILE_END(timeNormalize2, 1e3, REPS);

            // // Profiling verbose
            // log("  BPC(ori)=" + std::to_string(timeNormalize1) + "ms");
            // log("  BPC(new)=" + std::to_string(timeNormalize2) + "ms");

            // Sanity check
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsGpu(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float* const heatMapGpuPtr,
        const float* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const float interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
        const float minSubsetScore, const float scaleFactor, const bool maximizePositives,
        Array<float> pairScoresCpu, float* pairScoresGpuPtr, const unsigned int* const bodyPartPairsGpuPtr,
        const unsigned int* const mapIdxGpuPtr, const float* const peaksGpuPtr);
    template void connectBodyPartsGpu(
        Array<double>& poseKeypoints, Array<double>& poseScores, const double* const heatMapGpuPtr,
        const double* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const double interMinAboveThreshold, const double interThreshold, const int minSubsetCnt,
        const double minSubsetScore, const double scaleFactor, const bool maximizePositives,
        Array<double> pairScoresCpu, double* pairScoresGpuPtr, const unsigned int* const bodyPartPairsGpuPtr,
        const unsigned int* const mapIdxGpuPtr, const double* const peaksGpuPtr);
}
