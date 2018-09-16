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
            const auto vectorAToBX = candidateBPtr[3*j] - candidateAPtr[3*i];
            const auto vectorAToBY = candidateBPtr[3*j+1] - candidateAPtr[3*i+1];
            const auto vectorAToBMax = fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
            const auto numberPointsInLine = fastMax(
                5, fastMin(25, intRound(std::sqrt(5*vectorAToBMax))));
            const auto vectorNorm = T(std::sqrt( vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY ));
            // If the peaksPtr are coincident. Don't connect them.
            if (vectorNorm > 1e-6)
            {
                const auto sX = candidateAPtr[3*i];
                const auto sY = candidateAPtr[3*i+1];
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
        const unsigned int numberBodyPartPairs, const unsigned int subsetCounterIndex, const Array<T>& precomputedPAFs)
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
                                        if (heatMapPtr == nullptr)
                                            error("HeatMapPtr is null. GPU PAF not implemented for star architecture.",
                                                  __LINE__, __FUNCTION__, __FILE__);
                                        const auto pairIndex2 = bodyPartPairsStar[bodyPartB];
                                        const auto* mapX0 = heatMapPtr
                                                          + (numberBodyPartsAndBkg + pairIndex2) * heatMapOffset;
                                        const auto* mapY0 = heatMapPtr
                                                          + (numberBodyPartsAndBkg + pairIndex2+1) * heatMapOffset;
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
                    // Note: Problem of this function, if no right PAF between A and B, both elements are
                    // discarded. However, they should be added indepently, not discarded
                    if (heatMapPtr != nullptr)
                    {
                        const auto* mapX = heatMapPtr
                                         + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                        const auto* mapY = heatMapPtr
                                         + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
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
                    else if (!precomputedPAFs.empty())
                    {
                        for (auto i = 1; i <= numberA; i++)
                        {
                            // E.g. neck-nose connection. For each nose
                            for (auto j = 1; j <= numberB; j++)
                            {
                                const auto scoreAB = precomputedPAFs.at(
                                    {(int)pairIndex, i-1, j-1});

                                // E.g. neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
                            }
                        }
                    }
                    else
                        error("Error. Should not reach here.", __LINE__, __FUNCTION__, __FILE__);

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
                                      const int minSubsetCnt, const T minSubsetScore, const int maxPeaks)
    {
        try
        {
            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            numberPeople = 0;
            validSubsetIndexes.clear();
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, subsets.size()));
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
                    if (numberPeople == maxPeaks)
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
    void connectDistanceStar(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T scaleFactor, const unsigned int numberBodyParts,
                             const unsigned int bodyPartPairsSize)
    {
        try
        {
            // poseKeypoints from neck-part distances
            if (poseModel == PoseModel::BODY_25D)
            {
                const auto scaleDownFactor = 8;
                Array<T> poseKeypoints2 = poseKeypoints.clone();
                const auto rootIndex = 1;
                const auto rootNumberIndex = rootIndex*(maxPeaks+1)*3;
                const auto numberPeople = intRound(peaksPtr[rootNumberIndex]);
                poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3}, 0);
                poseScores.reset(numberPeople, 0);
                // // 48 channels
                // const std::vector<float> AVERAGE{
                //     0.f, -2.76364f, -1.3345f, 0.f,   -1.95322f, 3.95679f, -1.20664f, 4.76543f,
                //     1.3345f, 0.f, 1.92318f, 3.96891f,   1.17999f, 4.7901f, 0.f, 7.72201f,
                //     -0.795236f, 7.74017f, -0.723963f,   11.209f, -0.651316f, 15.6972f,
                //     0.764623f, 7.74869f, 0.70755f,   11.2307f, 0.612832f, 15.7281f,
                //     -0.123134f, -3.43515f,   0.111775f, -3.42761f,
                //     -0.387066f, -3.16603f,   0.384038f, -3.15951f,
                //     0.344764f, 12.9666f, 0.624157f,   12.9057f, 0.195454f, 12.565f,
                //     -1.06074f, 12.9951f, -1.2427f,   12.9309f, -0.800837f, 12.5845f};
                // const std::vector<float> SIGMA{
                //     3.39629f, 3.15605f, 3.16913f, 1.8234f,   5.82252f, 5.05674f, 7.09876f, 6.64574f,
                //     3.16913f, 1.8234f, 5.79415f, 5.01424f,   7.03866f, 6.62427f, 5.52593f, 6.75962f,
                //     5.91224f, 6.87241f, 8.66473f,   10.1792f, 11.5871f, 13.6565f,
                //     5.86653f, 6.89568f, 8.68067f,   10.2127f, 11.5954f, 13.6722f,
                //     3.3335f, 3.49128f,   3.34476f, 3.50079f,
                //     2.93982f, 3.11151f,   2.95006f, 3.11004f,
                //     9.69408f, 7.58921f, 9.71193f,   7.44185f, 9.19343f, 7.11157f,
                //     9.16848f, 7.86122f, 9.07613f,   7.83682f, 8.91951f, 7.33715f};
                // 50 channels
                const std::vector<float> AVERAGE{
                    0, -6.55251,
                    0, -4.15062, -1.48818, -4.15506,   -2.22408, -0.312264, -1.42204, 0.588495,
                    1.51044, -4.14629, 2.2113, -0.312283,   1.41081, 0.612377, -0, 3.41112,
                    -0.932306, 3.45504, -0.899812,   6.79837, -0.794223, 11.4972,
                    0.919047, 3.46442, 0.902314,   6.81245, 0.79518, 11.5132,
                    -0.243982, -7.07925,   0.28065, -7.07398,
                    -0.792812, -7.09374,   0.810145, -7.06958,
                    0.582387, 7.46846, 0.889349,   7.40577, 0.465088, 7.03969,
                    -0.96686, 7.46148, -1.20773,   7.38834, -0.762135, 6.99575};
                const std::vector<float> SIGMA{
                    7.26789, 9.70751,
                    6.29588, 8.93472, 6.97401, 9.13746,   7.49632, 9.44757, 8.06695, 9.97319,
                    6.99726, 9.14608, 7.50529, 9.43568,   8.05888, 9.98207, 6.38929, 9.29314,
                    6.71801, 9.39271, 8.00608,   10.6141, 10.3416, 12.7812,
                    6.69875, 9.41407, 8.01876,   10.637, 10.3475, 12.7849,
                    7.30923, 9.7324,   7.27886, 9.73406,
                    7.35978, 9.7289,   7.28914, 9.67711,
                    7.93153, 8.10845, 7.95577,   8.01729, 7.56865, 7.87314,
                    7.4655, 8.25336, 7.43958,   8.26333, 7.33667, 7.97446};
                // To get ideal distance
                const auto numberBodyPartsAndBkgAndPAFChannels = numberBodyParts + 1 + bodyPartPairsSize;
                const auto heatMapOffset = heatMapSize.area();
                // For each person
                for (auto p = 0 ; p < numberPeople ; p++)
                {
                    // For root (neck) position
                    // bpOrig == rootIndex
                    const auto rootXYSIndex = rootNumberIndex+3*(1+p);
                    // Set (x,y,score)
                    const auto rootX = scaleFactor*peaksPtr[rootXYSIndex];
                    const auto rootY = scaleFactor*peaksPtr[rootXYSIndex+1];
                    poseKeypoints[{p,rootIndex,0}] = rootX;
                    poseKeypoints[{p,rootIndex,1}] = rootY;
                    poseKeypoints[{p,rootIndex,2}] = peaksPtr[rootXYSIndex+2];
                    // For each body part
                    for (auto bpOrig = 0 ; bpOrig < (int)numberBodyParts ; bpOrig++)
                    {
                        if (bpOrig != rootIndex)
                        {
                            // // 48 channels
                            // const auto bpChannel = (bpOrig < rootIndex ? bpOrig : bpOrig-1);
                            // 50 channels
                            const auto bpChannel = bpOrig;
                            // Get ideal distance
                            const auto offsetIndex = numberBodyPartsAndBkgAndPAFChannels + 2*bpChannel;
                            const auto* mapX = heatMapPtr + offsetIndex * heatMapOffset;
                            const auto* mapY = heatMapPtr + (offsetIndex+1) * heatMapOffset;
                            const auto increaseRatio = scaleFactor*scaleDownFactor;
                            // Set (x,y) coordinates from the distance
                            const auto indexChannel = 2*bpChannel;
                            // // Not refined method
                            // const auto index = intRound(
                            //     rootY/scaleFactor)*heatMapSize.x + intRound(rootX/scaleFactor);
                            // const Point<T> neckPartDist{
                            //     increaseRatio*(mapX[index]*SIGMA[indexChannel]+AVERAGE[indexChannel]),
                            //     increaseRatio*(mapY[index]*SIGMA[indexChannel+1]+AVERAGE[indexChannel+1])};
                            // poseKeypoints[{p,bpOrig,0}] = rootX + neckPartDist.x;
                            // poseKeypoints[{p,bpOrig,1}] = rootY + neckPartDist.y;
                            // Refined method
                            const auto constant = 5;
                            Point<T> neckPartDistRefined{0, 0};
                            auto counterRefinements = 0;
                            // We must keep it inside the image size
                            for (auto y = fastMax(0, intRound(rootY/scaleFactor) - constant);
                                 y < fastMin(heatMapSize.y, intRound(rootY/scaleFactor) + constant+1) ; y++)
                            {
                                for (auto x = fastMax(0, intRound(rootX/scaleFactor) - constant);
                                     x < fastMin(heatMapSize.x, intRound(rootX/scaleFactor) + constant+1) ; x++)
                                {
                                    const auto index = y*heatMapSize.x + x;
                                    neckPartDistRefined.x += mapX[index];
                                    neckPartDistRefined.y += mapY[index];
                                    counterRefinements++;
                                }
                            }
                            neckPartDistRefined = Point<T>{
                                neckPartDistRefined.x*SIGMA[indexChannel]+counterRefinements*AVERAGE[indexChannel],
                                neckPartDistRefined.y*SIGMA[indexChannel+1]+counterRefinements*AVERAGE[indexChannel+1],
                            };
                            neckPartDistRefined *= increaseRatio/counterRefinements;
                            const auto partX = rootX + neckPartDistRefined.x;
                            const auto partY = rootY + neckPartDistRefined.y;
                            poseKeypoints[{p,bpOrig,0}] = partX;
                            poseKeypoints[{p,bpOrig,1}] = partY;
                            // Set (temporary) body part score
                            poseKeypoints[{p,bpOrig,2}] = T(0.0501);
                            // Associate estimated keypoint with closest one
                            const auto xCleaned = fastMax(0, fastMin(heatMapSize.x-1, intRound(partX/scaleFactor)));
                            const auto yCleaned = fastMax(0, fastMin(heatMapSize.y-1, intRound(partY/scaleFactor)));
                            const auto partConfidence = heatMapPtr[
                                bpOrig * heatMapOffset + yCleaned*heatMapSize.x + xCleaned];
                            // If partConfidence is big enough, it means we are close to a keypoint
                            if (partConfidence > T(0.05))
                            {
                                const auto candidateNumberIndex = bpOrig*(maxPeaks+1)*3;
                                const auto numberCandidates = intRound(peaksPtr[candidateNumberIndex]);
                                int closestIndex = -1;
                                T closetValue = std::numeric_limits<T>::max();
                                for (auto i = 0 ; i < numberCandidates ; i++)
                                {
                                    const auto candidateXYSIndex = candidateNumberIndex+3*(1+i);
                                    const auto diffX = partX-scaleFactor*peaksPtr[candidateXYSIndex];
                                    const auto diffY = partY-scaleFactor*peaksPtr[candidateXYSIndex+1];
                                    const auto dist = (diffX*diffX + diffY*diffY);
                                    if (closetValue > dist)
                                    {
                                        closetValue = dist;
                                        closestIndex = candidateXYSIndex;
                                    }
                                }
                                if (closestIndex != -1)
                                {
                                    poseKeypoints[{p,bpOrig,0}] = scaleFactor*peaksPtr[closestIndex];
                                    poseKeypoints[{p,bpOrig,1}] = scaleFactor*peaksPtr[closestIndex+1];
                                    // Set body part score
                                    poseKeypoints[{p,bpOrig,2}] = peaksPtr[closestIndex+2];
                                }
                            }
                            // Set poseScore
                            poseScores[p] += poseKeypoints[{p,bpOrig,2}];
                        }
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const std::vector<float> AVERAGE{
        0, -6.55251,
        0, -4.15062, -1.48818, -4.15506,   -2.22408, -0.312264, -1.42204, 0.588495,
        1.51044, -4.14629, 2.2113, -0.312283,   1.41081, 0.612377, -0, 3.41112,
        -0.932306, 3.45504, -0.899812,   6.79837, -0.794223, 11.4972,
        0.919047, 3.46442, 0.902314,   6.81245, 0.79518, 11.5132,
        -0.243982, -7.07925,   0.28065, -7.07398,
        -0.792812, -7.09374,   0.810145, -7.06958,
        0.582387, 7.46846, 0.889349,   7.40577, 0.465088, 7.03969,
        -0.96686, 7.46148, -1.20773,   7.38834, -0.762135, 6.99575};
    const std::vector<float> SIGMA{
        7.26789, 9.70751,
        6.29588, 8.93472, 6.97401, 9.13746,   7.49632, 9.44757, 8.06695, 9.97319,
        6.99726, 9.14608, 7.50529, 9.43568,   8.05888, 9.98207, 6.38929, 9.29314,
        6.71801, 9.39271, 8.00608,   10.6141, 10.3416, 12.7812,
        6.69875, 9.41407, 8.01876,   10.637, 10.3475, 12.7849,
        7.30923, 9.7324,   7.27886, 9.73406,
        7.35978, 9.7289,   7.28914, 9.67711,
        7.93153, 8.10845, 7.95577,   8.01729, 7.56865, 7.87314,
        7.4655, 8.25336, 7.43958,   8.26333, 7.33667, 7.97446};
    template <typename T>
    std::array<T,3> regressPart(const Array<T>& person, const int rootIndex, const int targetIndex,
                                const int scaleDownFactor, const T* const heatMapPtr, const T* const peaksPtr,
                                const Point<int>& heatMapSize, const int maxPeaks, const T scaleFactor,
                                const unsigned int numberBodyPartsAndBkgAndPAFChannels)
    {
        try
        {
            std::array<T,3> result{0,0,0};
            // poseKeypoints from neck-part distances
            if (targetIndex != rootIndex && person[{rootIndex,2}] > T(0.05))
            {
                // Set (x,y)
                const auto rootX = person[{rootIndex,0}];
                const auto rootY = person[{rootIndex,1}];
                // Get ideal distance
                const auto indexChannel = 2*targetIndex;
                const auto offsetIndex = numberBodyPartsAndBkgAndPAFChannels + indexChannel;
                const auto heatMapOffset = heatMapSize.area();
                const auto* mapX = heatMapPtr + offsetIndex * heatMapOffset;
                const auto* mapY = heatMapPtr + (offsetIndex+1) * heatMapOffset;
                const auto increaseRatio = scaleFactor*scaleDownFactor;
                // // Not refined method
                // const auto index = intRound(rootY/scaleFactor)*heatMapSize.x + intRound(rootX/scaleFactor);
                // const Point<T> neckPartDist{
                //     increaseRatio*(mapX[index]*SIGMA[indexChannel]+AVERAGE[indexChannel]),
                //     increaseRatio*(mapY[index]*SIGMA[indexChannel+1]+AVERAGE[indexChannel+1])};
                // poseKeypoints[{p,targetIndex,0}] = rootX + neckPartDist.x;
                // poseKeypoints[{p,targetIndex,1}] = rootY + neckPartDist.y;
                // Refined method
                const auto constant = 5;
                Point<T> neckPartDistRefined{0, 0};
                auto counterRefinements = 0;
                // We must keep it inside the image size
                for (auto y = fastMax(0, intRound(rootY/scaleFactor) - constant);
                     y < fastMin(heatMapSize.y, intRound(rootY/scaleFactor) + constant+1) ; y++)
                {
                    for (auto x = fastMax(0, intRound(rootX/scaleFactor) - constant);
                         x < fastMin(heatMapSize.x, intRound(rootX/scaleFactor) + constant+1) ; x++)
                    {
                        const auto index = y*heatMapSize.x + x;
                        neckPartDistRefined.x += mapX[index];
                        neckPartDistRefined.y += mapY[index];
                        counterRefinements++;
                    }
                }
                neckPartDistRefined = Point<T>{
                    neckPartDistRefined.x*SIGMA[indexChannel]+counterRefinements*AVERAGE[indexChannel],
                    neckPartDistRefined.y*SIGMA[indexChannel+1]+counterRefinements*AVERAGE[indexChannel+1],
                };
                neckPartDistRefined *= increaseRatio/counterRefinements;
                const auto partX = rootX + neckPartDistRefined.x;
                const auto partY = rootY + neckPartDistRefined.y;
                result[0] = partX;
                result[1] = partY;
                // Set (temporary) body part score
                result[2] = T(0.0501);
                // Associate estimated keypoint with closest one
                const auto xCleaned = fastMax(0, fastMin(heatMapSize.x-1, intRound(partX/scaleFactor)));
                const auto yCleaned = fastMax(0, fastMin(heatMapSize.y-1, intRound(partY/scaleFactor)));
                const auto partConfidence = heatMapPtr[
                    targetIndex * heatMapOffset + yCleaned*heatMapSize.x + xCleaned];
                // If partConfidence is big enough, it means we are close to a keypoint
                if (partConfidence > T(0.05))
                {
                    const auto candidateNumberIndex = targetIndex*(maxPeaks+1)*3;
                    const auto numberCandidates = intRound(peaksPtr[candidateNumberIndex]);
                    int closestIndex = -1;
                    T closetValue = std::numeric_limits<T>::max();
                    for (auto i = 0 ; i < numberCandidates ; i++)
                    {
                        const auto candidateXYSIndex = candidateNumberIndex+3*(1+i);
                        const auto diffX = partX-scaleFactor*peaksPtr[candidateXYSIndex];
                        const auto diffY = partY-scaleFactor*peaksPtr[candidateXYSIndex+1];
                        const auto dist = (diffX*diffX + diffY*diffY);
                        if (closetValue > dist)
                        {
                            closetValue = dist;
                            closestIndex = candidateXYSIndex;
                        }
                    }
                    if (closestIndex != -1)
                    {
                        result[0] = scaleFactor*peaksPtr[closestIndex];
                        result[1] = scaleFactor*peaksPtr[closestIndex+1];
                        // Set body part score
                        result[2] = peaksPtr[closestIndex+2];
                    }
                }
            }
            return result;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::array<T,3>{};
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
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, subsets.size()));
            removeSubsetsBelowThresholds(validSubsetIndexes, numberPeople, subsets, subsetCounterIndex,
                                         numberBodyParts, minSubsetCnt, minSubsetScore, maxPeaks);

            // Fill and return poseKeypoints
            subsetsToPoseKeypointsAndScores(poseKeypoints, poseScores, scaleFactor, subsets, validSubsetIndexes,
                                            peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);

            // poseKeypoints from neck-part distances
            if (poseModel == PoseModel::BODY_25D)
//                 connectDistanceStar(poseKeypoints, poseScores, heatMapPtr, peaksPtr, poseModel, heatMapSize,
//                                     maxPeaks, scaleFactor, numberBodyParts, bodyPartPairs.size());
// if (false)
            {
                // Add all the root elements (necks)
                const std::vector<int> keypointsSize = {(int)numberBodyParts, 3};
                // Initial #people = number root elements
                const auto rootIndex = 1;
                std::vector<Array<T>> poseKeypointsTemp;
                // Iterate for each body part
                const std::array<int, 25> MAPPING{
                    1, 8, 0, 2,5,9,12, 3,6,10,13, 15,16, 4,7,11,14, 17,18, 19,22,20,23,21,24};
                const auto numberBodyPartsAndBkgAndPAFChannels = numberBodyParts + 1 + bodyPartPairs.size();
                const auto scaleDownFactor = 8;
                for (auto index = 0u ; index < numberBodyParts ; index++)
                {
                    const auto targetIndex = MAPPING[index];
                    // Get all candidate keypoints
                    const auto partNumberIndex = targetIndex*(maxPeaks+1)*3;
                    const auto numberPartParts = intRound(peaksPtr[partNumberIndex]);
                    std::vector<std::array<T, 3>> currentPartCandidates(numberPartParts);
                    for (auto i = 0u ; i < currentPartCandidates.size() ; i++)
                    {
                        const auto baseIndex = partNumberIndex+3*(i+1);
                        currentPartCandidates[i][0] = scaleFactor*peaksPtr[baseIndex];
                        currentPartCandidates[i][1] = scaleFactor*peaksPtr[baseIndex+1];
                        currentPartCandidates[i][2] = peaksPtr[baseIndex+2];
                    }
                    // Detect new body part for existing people
                    // For each temporary person --> Add new targetIndex part
                    for (auto& person : poseKeypointsTemp)
                    {
                        // Estimate new body part w.r.t. each already-detected body part
                        for (auto rootMapIndex = 0u ; rootMapIndex < index ; rootMapIndex++)
                        {
                            const auto rootIndex = MAPPING[rootMapIndex];
                            if (person[{rootIndex,2}] > T(0.0501))
                            {
                                const auto result = regressPart(
                                    person, rootIndex, targetIndex, scaleDownFactor, heatMapPtr, peaksPtr,
                                    heatMapSize, maxPeaks, scaleFactor, numberBodyPartsAndBkgAndPAFChannels);
                                if (person[{targetIndex,2}] < result[2])
                                {
                                    person[{targetIndex,0}] = result[0];
                                    person[{targetIndex,1}] = result[1];
                                    person[{targetIndex,2}] = result[2];
                                }
                            }
                        }
                    }
                    // Add leftovers body parts as new people
if (targetIndex == rootIndex)
{
                    const auto currentSize = poseKeypointsTemp.size();
                    poseKeypointsTemp.resize(currentSize+currentPartCandidates.size());
                    for (auto p = 0u ; p < currentPartCandidates.size() ; p++)
                    {
                        poseKeypointsTemp[currentSize+p] = Array<T>(keypointsSize, 0.f);
                        const auto baseIndex = 3*targetIndex;
                        poseKeypointsTemp[currentSize+p][baseIndex  ] = currentPartCandidates[p][0];
                        poseKeypointsTemp[currentSize+p][baseIndex+1] = currentPartCandidates[p][1];
                        poseKeypointsTemp[currentSize+p][baseIndex+2] = currentPartCandidates[p][2];
                    }

}
                }
                // poseKeypoints: Reformat poseKeypointsTemp as poseKeypoints
                poseKeypoints.reset({(int)poseKeypointsTemp.size(), (int)numberBodyParts, 3}, 0);
                poseScores.reset(poseKeypoints.getSize(0), 0.f);
                const auto keypointArea = poseKeypoints.getSize(1)*poseKeypoints.getSize(2);
                for (auto p = 0 ; p < poseKeypoints.getSize(0) ; p++)
                {
                    const auto pIndex = p*keypointArea;
                    for (auto part = 0 ; part < poseKeypoints.getSize(1) ; part++)
                    {
                        const auto baseIndexTemp = 3*part;
                        const auto baseIndex = pIndex+baseIndexTemp;
                        poseKeypoints[baseIndex  ] = poseKeypointsTemp[p][baseIndexTemp];
                        poseKeypoints[baseIndex+1] = poseKeypointsTemp[p][baseIndexTemp+1];
                        poseKeypoints[baseIndex+2] = poseKeypointsTemp[p][baseIndexTemp+2];
                        // Set poseScore
                        poseScores[p] += poseKeypoints[baseIndex+2];
                    }
                }
            }
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
