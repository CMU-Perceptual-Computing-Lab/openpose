#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>

namespace op
{
    template <typename T>
    inline T getScoreAB(const int i, const int j, const T* const candidateAPtr, const T* const candidateBPtr,
                        const T* const mapX, const T* const mapY, const Point<int>& heatMapSize,
                        const T interThreshold, const T interMinAboveThreshold)
    {
        try
        {
            const auto vectorAToBX = candidateBPtr[j*3] - candidateAPtr[i*3];
            const auto vectorAToBY = candidateBPtr[j*3+1] - candidateAPtr[i*3+1];
            const auto vectorAToBMax = fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
            const auto numberPointsInLine = fastMax(
                5, fastMin(25, intRound(std::sqrt(5*vectorAToBMax))));
            const auto vectorNorm = T(std::sqrt( vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY ));
            // If the peaksPtr are coincident. Don't connect them.
            if (vectorNorm > 1e-6)
            {
                const auto sX = candidateAPtr[i*3];
                const auto sY = candidateAPtr[i*3+1];
                const auto vectorAToBNormX = vectorAToBX/vectorNorm;
                const auto vectorAToBNormY = vectorAToBY/vectorNorm;

                auto sum = T(0);
                auto count = 0u;
                const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
                const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
                for (auto lm = 0; lm < numberPointsInLine; lm++)
                {
                    const auto mX = fastMax(
                        0, fastMin(heatMapSize.x-1, intRound(sX + lm*vectorAToBXInLine)));
                    const auto mY = fastMax(
                        0, fastMin(heatMapSize.y-1, intRound(sY + lm*vectorAToBYInLine)));
                    const auto idx = mY * heatMapSize.x + mX;
                    const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                    if (score > interThreshold)
                    {
                        sum += score;
                        count++;
                    }
                }
                if (count/(float)numberPointsInLine > interMinAboveThreshold)
                    return sum/count;
            }
            return T(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }
    template <typename T>
    inline T getScore0B(const int bodyPart0, const T* const candidate0Ptr, const int i, const int j,
                        const int bodyPartA, const int bodyPartB, const T* const candidateBPtr,
                        const T* const heatMapPtr, const Point<int>& heatMapSize,
                        const T interThreshold, const T interMinAboveThreshold, const int peaksOffset,
                        const int heatMapOffset, const int numberBodyPartsAndBkg,
                        const std::vector<std::pair<std::vector<int>, double>>& subsets,
                        const std::vector<int>& bodyPartPairsStar)
    {
        try
        {
            // A is already in the subsets, find its connection B
            const auto pairIndex2 = bodyPartPairsStar[bodyPartB];
            const auto* mapX0 = heatMapPtr + (numberBodyPartsAndBkg + pairIndex2) * heatMapOffset;
            const auto* mapY0 = heatMapPtr + (numberBodyPartsAndBkg + pairIndex2+1) * heatMapOffset;
            const int indexA = bodyPartA*peaksOffset + i*3 + 2;
            for (auto& subset : subsets)
            {
                const auto index0 = subset.first[bodyPart0];
                if (index0 > 0)
                {
                    // Found partA in a subsets, add partB to same one.
                    if (subset.first[bodyPartA] == indexA)
                    {
                        // index0 = std::get<0>(abConnection) = bodyPart0*peaksOffset + i0*3 + 2
                        // i0 = (index0 - 2 - bodyPart0*peaksOffset)/3
                        const auto i0 = (index0 - 2 - bodyPart0*peaksOffset)/3.;
                        return getScoreAB(i0, j, candidate0Ptr, candidateBPtr, mapX0, mapY0,
                                          heatMapSize, interThreshold, interMinAboveThreshold);
                    }
                }
            }
            return T(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }

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
            const auto& bodyPartPairsStar = getPosePartPairsStar(poseModel);
            // Star-PAF
            const auto bodyPart0 = 1;
            const auto* candidate0Ptr = peaksPtr + bodyPart0*peaksOffset;
            const auto number0 = intRound(candidate0Ptr[0]);
            // Iterate over it PAF connection, e.g. neck-nose, neck-Lshoulder, etc.
            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberA = intRound(candidateAPtr[0]);
                const auto numberB = intRound(candidateBPtr[0]);

                // E.g. neck-nose connection. If one of them is empty (e.g. no noses detected)
                // Add the non-empty elements into the subsets
                if (numberA == 0 || numberB == 0)
                {
                    // E.g. neck-nose connection. If no necks, add all noses
                    // Change w.r.t. other
                    if (numberA == 0) // numberB == 0 or not
                    {
                        // Non-MPI
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberB; i++)
                            {
                                bool found = false;
                                for (const auto& subset : subsets)
                                {
                                    const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                    if (subset.first[bodyPartB] == off)
                                    {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found)
                                {
                                    // Refinement: star-PAF
                                    // Look for root-B connection
                                    auto maxScore = T(0);
                                    auto maxScoreIndex = -1;
                                    if (poseModel == PoseModel::BODY_25E && bodyPartPairsStar[bodyPartB] > -1)
                                    {
                                        const auto pairIndex2 = bodyPartPairsStar[bodyPartB];
                                        const auto* mapX0 = heatMapPtr + (numberBodyPartsAndBkg + pairIndex2) * heatMapOffset;
                                        const auto* mapY0 = heatMapPtr + (numberBodyPartsAndBkg + pairIndex2+1) * heatMapOffset;
                                        for (auto j = 1; j <= number0; j++)
                                        {
                                            const auto score0B = getScoreAB(j, i, candidate0Ptr, candidateBPtr,
                                                                            mapX0, mapY0, heatMapSize, interThreshold,
                                                                            interMinAboveThreshold);
                                            if (maxScore < score0B)
                                            {
                                                maxScore = score0B;
                                                maxScoreIndex = j;
                                            }
                                        }
                                    }
                                    // Star-PAF --> found
                                    if (maxScore > 0)
                                    {
                                        // bool found = false;
                                        for (auto& subset : subsets)
                                        {
                                            const int index0 = bodyPart0*peaksOffset + maxScoreIndex*3 + 2;
                                            // Found partA in a subsets, add partB to same one.
                                            if (subset.first[bodyPart0] == index0)
                                            {
                                                const auto indexB = bodyPartB*peaksOffset + i*3 + 2;
                                                subset.first[bodyPartB] = indexB;
                                                subset.first[subsetCounterIndex]++;
                                                subset.second += peaksPtr[indexB] + maxScore;
                                                // found = true;
                                                break;
                                            }
                                        }
                                        // // TODO: This should never happen, but it does in 5K val
                                        // if (!found)
                                        // {
                                        //     const int index0 = bodyPart0*peaksOffset + maxScoreIndex*3 + 2;
                                        //     log(bodyPart0);
                                        //     log(index0);
                                        //     log(maxScoreIndex);
                                        //     log(maxScore);
                                        //     log(peaksPtr[index0]);
                                        //     error("Bug in this function if this happens. Report it to us.",
                                        //           __LINE__, __FUNCTION__, __FILE__);
                                        // }
                                    }
                                    // Add new subset with this element - Non-star-PAF code or Star-PAF when not found
                                    else // if (!found)
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
                        }
                        // MPI
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
                        // Non-MPI
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberA; i++)
                            {
                                bool found = false;
                                const auto indexA = bodyPartA;
                                for (const auto& subset : subsets)
                                {
                                    const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                    if (subset.first[indexA] == off)
                                    {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found)
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
                        // MPI
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
                    // (score, x, y). Inverted order for easy std::sort
                    std::vector<std::tuple<double, int, int>> allABConnections;
                    // Note: Problem of this function, if no right PAF between A and B, both elements are discarded.
                    // However, they should be added indepently, not discarded
                    {
                        const auto* mapX = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                        const auto* mapY = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
                        // E.g. neck-nose connection. For each neck
                        for (auto i = 1; i <= numberA; i++)
                        {
                            // E.g. neck-nose connection. For each nose
                            for (auto j = 1; j <= numberB; j++)
                            {
                                // Initial PAF
                                auto scoreAB = getScoreAB(i, j, candidateAPtr, candidateBPtr, mapX, mapY,
                                                          heatMapSize, interThreshold, interMinAboveThreshold);

                                // Refinement: star-PAF
                                if (poseModel == PoseModel::BODY_25E && bodyPartPairsStar[bodyPartB] > -1)
                                {
                                    const auto score0B = getScore0B(
                                        bodyPart0, candidate0Ptr, i, j, bodyPartA, bodyPartB, candidateBPtr,
                                        heatMapPtr, heatMapSize, interThreshold, interMinAboveThreshold, peaksOffset,
                                        heatMapOffset, numberBodyPartsAndBkg, subsets, bodyPartPairsStar);
                                    // Max
                                    scoreAB = fastMax(scoreAB, score0B);
                                    // // Smart average
                                    // // Both scores --> average
                                    // if (scoreAB > 0 && score0B > 0)
                                    // {
                                    //     const auto ratio0 = T(0.5);
                                    //     scoreAB = (1-ratio0)*scoreAB
                                    //             + ratio0 * score0B;
                                    // }
                                    // // No scoreAB --> use score0B
                                    // else if (score0B > 0)
                                    //     scoreAB = score0B;
                                    // // Else: No score0B --> use scoreAB
                                }

                                // E.g. neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
                            }
                        }
                    }

                    // select the top minAB connection, assuming that each part occur only once
                    // sort rows in descending order based on parts + connection score
                    if (!allABConnections.empty())
                        std::sort(allABConnections.begin(), allABConnections.end(),
                                  std::greater<std::tuple<double, int, int>>());

                    std::vector<std::tuple<int, int, double>> abConnections; // (x, y, score)
                    {
                        const auto minAB = fastMin(numberA, numberB);
                        std::vector<int> occurA(numberA, 0);
                        std::vector<int> occurB(numberB, 0);
                        auto counter = 0;
                        for (auto row = 0u; row < allABConnections.size(); row++)
                        {
                            const auto score = std::get<0>(allABConnections[row]);
                            const auto i = std::get<1>(allABConnections[row]);
                            const auto j = std::get<2>(allABConnections[row]);
                            if (!occurA[i-1] && !occurB[j-1])
                            {
                                abConnections.emplace_back(std::make_tuple(bodyPartA*peaksOffset + i*3 + 2,
                                                                           bodyPartB*peaksOffset + j*3 + 2,
                                                                           score));
                                counter++;
                                if (counter==minAB)
                                    break;
                                occurA[i-1] = 1;
                                occurB[j-1] = 1;
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
                        // Note: This has some issues:
                        //     - It does not prevent repeating the same keypoint in different people
                        //     - Assuming I have nose,eye,ear as 1 subset, and whole arm as another one, it will not
                        //       merge them both
                        else if (
                            (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                            || ((numberBodyParts == 19 || (numberBodyParts == 25 && poseModel != PoseModel::BODY_25E)
                                 || numberBodyParts == 59 || numberBodyParts == 65)
                                && (pairIndex==18 || pairIndex==19))
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
                                    {
                                        subsetB = indexB;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // subset.first[subsetCounterIndex]++;
                                    }
                                    else if (subsetB == indexB && subsetA == 0)
                                    {
                                        subsetA = indexA;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // subset.first[subsetCounterIndex]++;
                                    }
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
                                bool found = false;
                                for (auto& subset : subsets)
                                {
                                    // Found partA in a subsets, add partB to same one.
                                    if (subset.first[bodyPartA] == indexA)
                                    {
                                        subset.first[bodyPartB] = indexB;
                                        subset.first[subsetCounterIndex]++;
                                        subset.second += peaksPtr[indexB] + score;
                                        found = true;
                                        break;
                                    }
                                }
                                // Not found partA in subsets, add new subsets element
                                if (!found)
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
