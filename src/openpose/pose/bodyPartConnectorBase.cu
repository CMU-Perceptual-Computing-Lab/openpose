#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>

namespace op
{
    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const T* const heatMapGpuPtr, const T* const peaksGpuPtr)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

            // Vector<int> = Each body part + body parts counter; double = subsetScore
            std::vector<std::pair<std::vector<int>, double>> subset;
            const auto subsetCounterIndex = numberBodyParts;
            const auto subsetSize = numberBodyParts+1;

            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.area();

            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberA = intRound(candidateAPtr[0]);
                const auto numberB = intRound(candidateBPtr[0]);

                // Add parts into the subset in special case
                if (numberA == 0 || numberB == 0)
                {
                    // Change w.r.t. other
                    if (numberA == 0) // numberB == 0 or not
                    {
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberB; i++)
                            {
                                bool num = false;
                                const auto indexB = bodyPartB;
                                for (auto j = 0u; j < subset.size(); j++)
                                {
                                    const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                    if (subset[j].first[indexB] == off)
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
                                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
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
                                subset.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                    else // if (numberA != 0 && numberB == 0)
                    {
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberA; i++)
                            {
                                bool num = false;
                                const auto indexA = bodyPartA;
                                for (auto j = 0u; j < subset.size(); j++)
                                {
                                    const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                    if (subset[j].first[indexA] == off)
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
                                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
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
                                subset.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                }
                else // if (numberA != 0 && numberB != 0)
                {
                    std::vector<std::tuple<double, int, int>> temp;
                    const auto* mapX = heatMapPtr + mapIdx[2*pairIndex] * heatMapOffset;
                    const auto* mapY = heatMapPtr + mapIdx[2*pairIndex+1] * heatMapOffset;
                    for (auto i = 1; i <= numberA; i++)
                    {
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
                                    const auto mX = fastMin(heatMapSize.x-1, intRound(sX + lm*vectorAToBXInLine));
                                    const auto mY = fastMin(heatMapSize.y-1, intRound(sY + lm*vectorAToBYInLine));
                                    checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
                                    checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
                                    const auto idx = mY * heatMapSize.x + mX;
                                    const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                                    if (score > interThreshold)
                                    {
                                        sum += score;
                                        count++;
                                    }
                                }

                                // parts score + connection score
                                if (count/(float)numberPointsInLine > interMinAboveThreshold)
                                    temp.emplace_back(std::make_tuple(sum/count, i, j));
                            }
                        }
                    }

                    // select the top minAB connection, assuming that each part occur only once
                    // sort rows in descending order based on parts + connection score
                    if (!temp.empty())
                        std::sort(temp.begin(), temp.end(), std::greater<std::tuple<T, int, int>>());

                    std::vector<std::tuple<int, int, double>> connectionK;
                    const auto minAB = fastMin(numberA, numberB);
                    std::vector<int> occurA(numberA, 0);
                    std::vector<int> occurB(numberB, 0);
                    auto counter = 0;
                    for (auto row = 0u; row < temp.size(); row++)
                    {
                        const auto score = std::get<0>(temp[row]);
                        const auto x = std::get<1>(temp[row]);
                        const auto y = std::get<2>(temp[row]);
                        if (!occurA[x-1] && !occurB[y-1])
                        {
                            connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x*3 + 2,
                                                                     bodyPartB*peaksOffset + y*3 + 2,
                                                                     score));
                            counter++;
                            if (counter==minAB)
                                break;
                            occurA[x-1] = 1;
                            occurB[y-1] = 1;
                        }
                    }

                    // Cluster all the body part candidates into subset based on the part connection
                    if (!connectionK.empty())
                    {
                        // initialize first body part connection 15&16
                        if (pairIndex==0)
                        {
                            for (const auto connectionKI : connectionK)
                            {
                                std::vector<int> rowVector(numberBodyParts+3, 0);
                                const auto indexA = std::get<0>(connectionKI);
                                const auto indexB = std::get<1>(connectionKI);
                                const auto score = std::get<2>(connectionKI);
                                rowVector[bodyPartPairs[0]] = indexA;
                                rowVector[bodyPartPairs[1]] = indexB;
                                rowVector[subsetCounterIndex] = 2;
                                // add the score of parts and the connection
                                const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                subset.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                        // Add ears connections (in case person is looking to opposite direction to camera)
                        else if (
                            (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                            || ((numberBodyParts == 19 || numberBodyParts == 59)
                                && (pairIndex==18 || pairIndex==19))
                            || (numberBodyParts == 23 && (pairIndex==22 || pairIndex==23))
                            || (poseModel == PoseModel::BODY_19b
                                && (pairIndex == numberBodyPartPairs-1 || pairIndex == numberBodyPartPairs-2))
                            )
                        {
                            for (const auto& connectionKI : connectionK)
                            {
                                const auto indexA = std::get<0>(connectionKI);
                                const auto indexB = std::get<1>(connectionKI);
                                for (auto& subsetJ : subset)
                                {
                                    auto& subsetJFirst = subsetJ.first[bodyPartA];
                                    auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
                                    if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
                                        subsetJFirstPlus1 = indexB;
                                    else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
                                        subsetJFirst = indexA;
                                }
                            }
                        }
                        else
                        {
                            // A is already in the subset, find its connection B
                            for (const auto& connectionKI : connectionK)
                            {
                                const auto indexA = std::get<0>(connectionKI);
                                const auto indexB = std::get<1>(connectionKI);
                                const auto score = std::get<2>(connectionKI);
                                auto num = 0;
                                for (auto& subsetJ : subset)
                                {
                                    if (subsetJ.first[bodyPartA] == indexA)
                                    {
                                        subsetJ.first[bodyPartB] = indexB;
                                        num++;
                                        subsetJ.first[subsetCounterIndex] = subsetJ.first[subsetCounterIndex] + 1;
                                        subsetJ.second += peaksPtr[indexB] + score;
                                    }
                                }
                                // if can not find partA in the subset, create a new subset
                                if (num==0)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    rowVector[bodyPartA] = indexA;
                                    rowVector[bodyPartB] = indexB;
                                    rowVector[subsetCounterIndex] = 2;
                                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                    }
                }
            }

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            auto numberPeople = 0;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
            for (auto index = 0u ; index < subset.size() ; index++)
            {
                const auto subsetCounter = subset[index].first[subsetCounterIndex];
                const auto subsetScore = subset[index].second;
                if (subsetCounter >= minSubsetCnt && (subsetScore/subsetCounter) >= minSubsetScore)
                {
                    numberPeople++;
                    validSubsetIndexes.emplace_back(index);
                    if (numberPeople == POSE_MAX_PEOPLE)
                        break;
                }
                else if (subsetCounter < 1)
                    error("Bad subsetCounter. Bug in this function if this happens.",
                          __LINE__, __FUNCTION__, __FILE__);
            }

            // Fill and return poseKeypoints
            if (numberPeople > 0)
            {
                poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3});
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
                const auto& subsetPair = subset[validSubsetIndexes[person]];
                const auto& subsetI = subsetPair.first;
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
                    const auto bodyPartIndex = subsetI[bodyPart];
                    if (bodyPartIndex > 0)
                    {
                        poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex-2] * scaleFactor;
                        poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex-1] * scaleFactor;
                        poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
                    }
                    else
                    {
                        poseKeypoints[baseOffset] = 0.f;
                        poseKeypoints[baseOffset + 1] = 0.f;
                        poseKeypoints[baseOffset + 2] = 0.f;
                    }
                }
                poseScores[person] = subsetPair.second / (float)(numberBodyPartsAndPAFs);
            }

            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsGpu(Array<float>& poseKeypoints, Array<float>& poseScores,
                                      const float* const heatMapPtr, const float* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const float interMinAboveThreshold, const float interThreshold,
                                      const int minSubsetCnt, const float minSubsetScore, const float scaleFactor,
                                      const float* const heatMapGpuPtr, const float* const peaksGpuPtr);
    template void connectBodyPartsGpu(Array<double>& poseKeypoints, Array<double>& poseScores,
                                      const double* const heatMapPtr, const double* const peaksPtr,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const double interMinAboveThreshold, const double interThreshold,
                                      const int minSubsetCnt, const double minSubsetScore, const double scaleFactor,
                                      const double* const heatMapGpuPtr, const double* const peaksGpuPtr);
}
