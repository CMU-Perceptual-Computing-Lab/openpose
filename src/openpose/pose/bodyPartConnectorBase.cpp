#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>

namespace op
{
    template <typename T>
    std::vector<std::pair<std::vector<int>, double>> generateInitialSubsets(
        const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
        const int maxPeaks, const T interThreshold, const T interMinAboveThreshold,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs, const unsigned int subsetCounterIndex)
    {
        try
        {
            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: subset score
            std::vector<std::pair<std::vector<int>, double>> subsets;
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyPartsAndBkg = numberBodyParts + 1;
            const auto subsetSize = numberBodyParts+1;
            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.area();
            // Iterate over it PAF connection, e.g. neck-nose, neck-Lshoulder, etc.
            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberA = intRound(candidateAPtr[0]);
                const auto numberB = intRound(candidateBPtr[0]);

                // E.g. neck-nose connection. If one of them is empty (e.g. no noses de-tected)
                // Add the non-empty elements into the subsets
                if (numberA == 0 || numberB == 0)
                {
                    // E.g. neck-nose connection. If no necks, add all noses
                    // Change w.r.t. other
                    if (numberA == 0) // numberB == 0 or not
                    {
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberB; i++)
                            {
                                bool num = false;
                                const auto indexB = bodyPartB;
                                for (const auto& subset : subsets)
                                {
                                    const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                    if (subset.first[indexB] == off)
                                    {
                                        num = true;
                                        break;
                                    }
                                }
                                if (!num)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector[subsetCounterIndex] = 1;
                                    const auto subsetScore = candidateBPtr[i*3+2];
                                    // Second last number in each row is the total score
                                    subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                        else
                        {
                            for (auto i = 1; i <= numberB; i++)
                            {
                                std::vector<int> rowVector(subsetSize, 0);
                                // Store the index
                                rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector[subsetCounterIndex] = 1;
                                // Second last number in each row is the total score
                                const auto subsetScore = candidateBPtr[i*3+2];
                                subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                    // E.g. neck-nose connection. If no noses, add all necks
                    else // if (numberA != 0 && numberB == 0)
                    {
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberA; i++)
                            {
                                bool num = false;
                                const auto indexA = bodyPartA;
                                for (const auto& subset : subsets)
                                {
                                    const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                    if (subset.first[indexA] == off)
                                    {
                                        num = true;
                                        break;
                                    }
                                }
                                if (!num)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector[subsetCounterIndex] = 1;
                                    // Second last number in each row is the total score
                                    const auto subsetScore = candidateAPtr[i*3+2];
                                    subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                        else
                        {
                            for (auto i = 1; i <= numberA; i++)
                            {
                                std::vector<int> rowVector(subsetSize, 0);
                                // Store the index
                                rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector[subsetCounterIndex] = 1;
                                // Second last number in each row is the total score
                                const auto subsetScore = candidateAPtr[i*3+2];
                                subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                }
                // E.g. neck-nose connection. If necks and noses, look for maximums
                else // if (numberA != 0 && numberB != 0)
                {
                    std::vector<std::tuple<double, int, int>> allABConnections; // (score, x, y). Inverted order for easy std::sort
                    {
                        const auto* mapX = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                        const auto* mapY = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
                        // E.g. neck-nose connection. For each neck
                        for (auto i = 1; i <= numberA; i++)
                        {
                            // E.g. neck-nose connection. For each nose
                            for (auto j = 1; j <= numberB; j++)
                            {
                                const auto vectorAToBX = candidateBPtr[j*3] - candidateAPtr[i*3];
                                const auto vectorAToBY = candidateBPtr[j*3+1] - candidateAPtr[i*3+1];
                                const auto vectorAToBMax = fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
                                const auto numberPointsInLine = fastMax(5, fastMin(25, intRound(std::sqrt(5*vectorAToBMax))));
                                const auto vectorNorm = T(std::sqrt( vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY ));
                                // If the peaksPtr are coincident. Don't connect them.
                                if (vectorNorm > 1e-6)
                                {
                                    const auto sX = candidateAPtr[i*3];
                                    const auto sY = candidateAPtr[i*3+1];
                                    const auto vectorAToBNormX = vectorAToBX/vectorNorm;
                                    const auto vectorAToBNormY = vectorAToBY/vectorNorm;

                                    auto sum = 0.;
                                    auto count = 0;
                                    const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
                                    const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
                                    for (auto lm = 0; lm < numberPointsInLine; lm++)
                                    {
                                        const auto mX = fastMax(0, fastMin(heatMapSize.x-1, intRound(sX + lm*vectorAToBXInLine)));
                                        const auto mY = fastMax(0, fastMin(heatMapSize.y-1, intRound(sY + lm*vectorAToBYInLine)));
                                        const auto idx = mY * heatMapSize.x + mX;
                                        const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                                        if (score > interThreshold)
                                        {
                                            sum += score;
                                            count++;
                                        }
                                    }

                                    // E.g. neck-nose connection. If possible PAF between neck i, nose j --> add
                                    // parts score + connection score
                                    if (count/(float)numberPointsInLine > interMinAboveThreshold)
                                        allABConnections.emplace_back(std::make_tuple(sum/count, i, j));
                                }
                            }
                        }
                    }

                    // select the top minAB connection, assuming that each part occur only once
                    // sort rows in descending order based on parts + connection score
                    if (!allABConnections.empty())
                        std::sort(allABConnections.begin(), allABConnections.end(), std::greater<std::tuple<double, int, int>>());

                    std::vector<std::tuple<int, int, double>> abConnections; // (x, y, score)
                    {
                        const auto minAB = fastMin(numberA, numberB);
                        std::vector<int> occurA(numberA, 0);
                        std::vector<int> occurB(numberB, 0);
                        auto counter = 0;
                        for (auto row = 0u; row < allABConnections.size(); row++)
                        {
                            const auto score = std::get<0>(allABConnections[row]);
                            const auto x = std::get<1>(allABConnections[row]);
                            const auto y = std::get<2>(allABConnections[row]);
                            if (!occurA[x-1] && !occurB[y-1])
                            {
                                abConnections.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x*3 + 2,
                                                                           bodyPartB*peaksOffset + y*3 + 2,
                                                                           score));
                                counter++;
                                if (counter==minAB)
                                    break;
                                occurA[x-1] = 1;
                                occurB[y-1] = 1;
                            }
                        }
                    }

                    // Cluster all the body part candidates into subsets based on the part connection
                    if (!abConnections.empty())
                    {
                        // initialize first body part connection 15&16
                        if (pairIndex==0)
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                std::vector<int> rowVector(numberBodyParts+3, 0);
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = std::get<2>(abConnection);
                                rowVector[bodyPartPairs[0]] = indexA;
                                rowVector[bodyPartPairs[1]] = indexB;
                                rowVector[subsetCounterIndex] = 2;
                                // add the score of parts and the connection
                                const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                        // Add ears connections (in case person is looking to opposite direction to camera)
                        else if (
                            (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                            || ((numberBodyParts == 19 || numberBodyParts == 25 || numberBodyParts == 59
                                 || numberBodyParts == 65)
                                && (pairIndex==18 || pairIndex==19))
                            || (poseModel == PoseModel::BODY_25E
                                && (pairIndex == numberBodyPartPairs-1 || pairIndex == numberBodyPartPairs-2))
                            )
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                for (auto& subset : subsets)
                                {
                                    auto& subsetA = subset.first[bodyPartA];
                                    auto& subsetB = subset.first[bodyPartB];
                                    if (subsetA == indexA && subsetB == 0)
                                        subsetB = indexB;
                                    else if (subsetB == indexB && subsetA == 0)
                                        subsetA = indexA;
                                }
                            }
                        }
                        else
                        {
                            // A is already in the subsets, find its connection B
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = std::get<2>(abConnection);
                                auto num = 0;
                                for (auto& subset : subsets)
                                {
                                    // Found partA in a subsets, add partB to same one.
                                    if (subset.first[bodyPartA] == indexA)
                                    {
                                        subset.first[bodyPartB] = indexB;
                                        num++;
                                        subset.first[subsetCounterIndex]++;
                                        subset.second += peaksPtr[indexB] + score;
                                    }
                                }
                                // Not found partA in subsets, add new subsets element
                                if (num==0)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    rowVector[bodyPartA] = indexA;
                                    rowVector[bodyPartB] = indexB;
                                    rowVector[subsetCounterIndex] = 2;
                                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                    subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                    }
                }
            }
            return subsets;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template <typename T>
    void removeSubsetsBelowThresholds(std::vector<int>& validSubsetIndexes, int& numberPeople,
                                      const std::vector<std::pair<std::vector<int>, double>>& subsets,
                                      const unsigned int subsetCounterIndex, const unsigned int numberBodyParts,
                                      const int minSubsetCnt, const T minSubsetScore)
    {
        try
        {
            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            numberPeople = 0;
            validSubsetIndexes.clear();
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subsets.size()));
            for (auto index = 0u ; index < subsets.size() ; index++)
            {
                auto subsetCounter = subsets[index].first[subsetCounterIndex];
                // Foot keypoints do not affect subsetCounter (too many false positives,
                // same foot usually appears as both left and right keypoints)
                // Pros: Removed tons of false positives
                // Cons: Standalone leg will never be recorded
                if (!COCO_CHALLENGE && numberBodyParts == 25)
                {
                    // No consider foot keypoints for that
                    for (auto i = 19 ; i < 25 ; i++)
                        subsetCounter -= (subsets[index].first.at(i) > 0);
                }
                const auto subsetScore = subsets[index].second;
                if (subsetCounter >= minSubsetCnt && (subsetScore/subsetCounter) >= minSubsetScore)
                {
                    numberPeople++;
                    validSubsetIndexes.emplace_back(index);
                    if (numberPeople == POSE_MAX_PEOPLE)
                        break;
                }
                else if ((subsetCounter < 1 && numberBodyParts != 25) || subsetCounter < 0)
                    error("Bad subsetCounter (" + std::to_string(subsetCounter) + "). Bug in this"
                          " function if this happens.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void subsetsToPoseKeypointsAndScores(Array<T>& poseKeypoints, Array<T>& poseScores, const T scaleFactor,
                                         const std::vector<std::pair<std::vector<int>, double>>& subsets,
                                         const std::vector<int>& validSubsetIndexes, const T* const peaksPtr,
                                         const int numberPeople, const unsigned int numberBodyParts,
                                         const unsigned int numberBodyPartPairs)
    {
        try
        {
            if (numberPeople > 0)
            {
                // Initialized to 0 for non-found keypoints in people
                poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3}, 0);
                poseScores.reset(numberPeople);
            }
            else
            {
                poseKeypoints.reset();
                poseScores.reset();
            }
            const auto numberBodyPartsAndPAFs = numberBodyParts + numberBodyPartPairs;
            for (auto person = 0u ; person < validSubsetIndexes.size() ; person++)
            {
                const auto& subsetPair = subsets[validSubsetIndexes[person]];
                const auto& subset = subsetPair.first;
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
                    const auto bodyPartIndex = subset[bodyPart];
                    if (bodyPartIndex > 0)
                    {
                        poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex-2] * scaleFactor;
                        poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex-1] * scaleFactor;
                        poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
                    }
                }
                poseScores[person] = subsetPair.second / (float)(numberBodyPartsAndPAFs);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void connectBodyPartsCpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
            const auto subsetCounterIndex = numberBodyParts;
            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: subset score
            const auto subsets = generateInitialSubsets(
                heatMapPtr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
                bodyPartPairs, numberBodyParts, numberBodyPartPairs, subsetCounterIndex);

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subsets.size()));
            removeSubsetsBelowThresholds(validSubsetIndexes, numberPeople, subsets, subsetCounterIndex,
                                         numberBodyParts, minSubsetCnt, minSubsetScore);

            // Fill and return poseKeypoints
            subsetsToPoseKeypointsAndScores(poseKeypoints, poseScores, scaleFactor, subsets, validSubsetIndexes,
                                            peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsCpu(Array<float>& poseKeypoints, Array<float>& poseScores,
                                      const float* const heatMapPtr, const float* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize,
                                      const int maxPeaks, const float interMinAboveThreshold,
                                      const float interThreshold, const int minSubsetCnt,
                                      const float minSubsetScore, const float scaleFactor);
    template void connectBodyPartsCpu(Array<double>& poseKeypoints, Array<double>& poseScores,
                                      const double* const heatMapPtr, const double* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize,
                                      const int maxPeaks, const double interMinAboveThreshold,
                                      const double interThreshold, const int minSubsetCnt,
                                      const double minSubsetScore, const double scaleFactor);
}
