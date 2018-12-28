#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>

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
                if (count/T(numberPointsInLine) > interMinAboveThreshold)
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
    std::vector<std::pair<std::vector<int>, T>> createPeopleVector(
        const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
        const int maxPeaks, const T interThreshold, const T interMinAboveThreshold,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs, const Array<T>& pairScores)
    {
        try
        {
            if (poseModel != PoseModel::BODY_25 && poseModel != PoseModel::COCO_18
                && poseModel != PoseModel::MPI_15 && poseModel != PoseModel::MPI_15_4)
                error("Model not implemented for CPU body connector.", __LINE__, __FUNCTION__, __FILE__);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: person subset score
            std::vector<std::pair<std::vector<int>, T>> peopleVector;
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyPartsAndBkg = numberBodyParts + (addBkgChannel(poseModel) ? 1 : 0);
            const auto vectorSize = numberBodyParts+1;
            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.area();
            // Iterate over it PAF connection, e.g., neck-nose, neck-Lshoulder, etc.
            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberPeaksA = intRound(candidateAPtr[0]);
                const auto numberPeaksB = intRound(candidateBPtr[0]);

                // E.g., neck-nose connection. If one of them is empty (e.g., no noses detected)
                // Add the non-empty elements into the peopleVector
                if (numberPeaksA == 0 || numberPeaksB == 0)
                {
                    // E.g., neck-nose connection. If no necks, add all noses
                    // Change w.r.t. other
                    if (numberPeaksA == 0) // numberPeaksB == 0 or not
                    {
                        // Non-MPI
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberPeaksB; i++)
                            {
                                bool found = false;
                                for (const auto& personVector : peopleVector)
                                {
                                    const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                    if (personVector.first[bodyPartB] == off)
                                    {
                                        found = true;
                                        break;
                                    }
                                }
                                // Add new personVector with this element
                                if (!found)
                                {
                                    std::vector<int> rowVector(vectorSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector.back() = 1;
                                    const auto personScore = candidateBPtr[i*3+2];
                                    // Second last number in each row is the total score
                                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                                }
                            }
                        }
                        // MPI
                        else
                        {
                            for (auto i = 1; i <= numberPeaksB; i++)
                            {
                                std::vector<int> rowVector(vectorSize, 0);
                                // Store the index
                                rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector.back() = 1;
                                // Second last number in each row is the total score
                                const auto personScore = candidateBPtr[i*3+2];
                                peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                            }
                        }
                    }
                    // E.g., neck-nose connection. If no noses, add all necks
                    else // if (numberPeaksA != 0 && numberPeaksB == 0)
                    {
                        // Non-MPI
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberPeaksA; i++)
                            {
                                bool found = false;
                                const auto indexA = bodyPartA;
                                for (const auto& personVector : peopleVector)
                                {
                                    const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                    if (personVector.first[indexA] == off)
                                    {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found)
                                {
                                    std::vector<int> rowVector(vectorSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector.back() = 1;
                                    // Second last number in each row is the total score
                                    const auto personScore = candidateAPtr[i*3+2];
                                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                                }
                            }
                        }
                        // MPI
                        else
                        {
                            for (auto i = 1; i <= numberPeaksA; i++)
                            {
                                std::vector<int> rowVector(vectorSize, 0);
                                // Store the index
                                rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector.back() = 1;
                                // Second last number in each row is the total score
                                const auto personScore = candidateAPtr[i*3+2];
                                peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                            }
                        }
                    }
                }
                // E.g., neck-nose connection. If necks and noses, look for maximums
                else // if (numberPeaksA != 0 && numberPeaksB != 0)
                {
                    // (score, indexA, indexB). Inverted order for easy std::sort
                    std::vector<std::tuple<double, int, int>> allABConnections;
                    // Note: Problem of this function, if no right PAF between A and B, both elements are
                    // discarded. However, they should be added indepently, not discarded
                    if (heatMapPtr != nullptr)
                    {
                        const auto* mapX = heatMapPtr
                                         + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                        const auto* mapY = heatMapPtr
                                         + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
                        // E.g., neck-nose connection. For each neck
                        for (auto i = 1; i <= numberPeaksA; i++)
                        {
                            // E.g., neck-nose connection. For each nose
                            for (auto j = 1; j <= numberPeaksB; j++)
                            {
                                // Initial PAF
                                auto scoreAB = getScoreAB(i, j, candidateAPtr, candidateBPtr, mapX, mapY,
                                                          heatMapSize, interThreshold, interMinAboveThreshold);

                                // E.g., neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
                            }
                        }
                    }
                    else if (!pairScores.empty())
                    {
                        const auto firstIndex = (int)pairIndex*pairScores.getSize(1)*pairScores.getSize(2);
                        // E.g., neck-nose connection. For each neck
                        for (auto i = 0; i < numberPeaksA; i++)
                        {
                            const auto iIndex = firstIndex + i*pairScores.getSize(2);
                            // E.g., neck-nose connection. For each nose
                            for (auto j = 0; j < numberPeaksB; j++)
                            {
                                const auto scoreAB = pairScores[iIndex + j];

                                // E.g., neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    // +1 because peaksPtr starts with counter
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i+1, j+1));
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
                        const auto minAB = fastMin(numberPeaksA, numberPeaksB);
                        std::vector<int> occurA(numberPeaksA, 0);
                        std::vector<int> occurB(numberPeaksB, 0);
                        auto counter = 0;
                        for (const auto& aBConnection : allABConnections)
                        {
                            const auto score = std::get<0>(aBConnection);
                            const auto indexA = std::get<1>(aBConnection);
                            const auto indexB = std::get<2>(aBConnection);
                            if (!occurA[indexA-1] && !occurB[indexB-1])
                            {
                                abConnections.emplace_back(std::make_tuple(bodyPartA*peaksOffset + indexA*3 + 2,
                                                                           bodyPartB*peaksOffset + indexB*3 + 2,
                                                                           score));
                                counter++;
                                if (counter==minAB)
                                    break;
                                occurA[indexA-1] = 1;
                                occurB[indexB-1] = 1;
                            }
                        }
                    }

                    // Cluster all the body part candidates into peopleVector based on the part connection
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
                                rowVector.back() = 2;
                                // add the score of parts and the connection
                                const auto personScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                            }
                        }
                        // Add ears connections (in case person is looking to opposite direction to camera)
                        // Note: This has some issues:
                        //     - It does not prevent repeating the same keypoint in different people
                        //     - Assuming I have nose,eye,ear as 1 person subset, and whole arm as another one, it will not
                        //       merge them both
                        else if (
                            (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                            || ((numberBodyParts == 19 || (numberBodyParts == 25)
                                 || numberBodyParts == 59 || numberBodyParts == 65)
                                && (pairIndex==18 || pairIndex==19))
                            )
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                for (auto& personVector : peopleVector)
                                {
                                    auto& personVectorA = personVector.first[bodyPartA];
                                    auto& personVectorB = personVector.first[bodyPartB];
                                    if (personVectorA == indexA && personVectorB == 0)
                                    {
                                        personVectorB = indexB;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // personVector.first.back()++;
                                    }
                                    else if (personVectorB == indexB && personVectorA == 0)
                                    {
                                        personVectorA = indexA;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // personVector.first.back()++;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // A is already in the peopleVector, find its connection B
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = T(std::get<2>(abConnection));
                                bool found = false;
                                for (auto& personVector : peopleVector)
                                {
                                    // Found partA in a peopleVector, add partB to same one.
                                    if (personVector.first[bodyPartA] == indexA)
                                    {
                                        personVector.first[bodyPartB] = indexB;
                                        personVector.first.back()++;
                                        personVector.second += peaksPtr[indexB] + score;
                                        found = true;
                                        break;
                                    }
                                }
                                // Not found partA in peopleVector, add new peopleVector element
                                if (!found)
                                {
                                    std::vector<int> rowVector(vectorSize, 0);
                                    rowVector[bodyPartA] = indexA;
                                    rowVector[bodyPartB] = indexB;
                                    rowVector.back() = 2;
                                    const auto personScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                                }
                            }
                        }
                    }
                }
            }
            return peopleVector;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template <typename T>
    std::vector<std::tuple<T, T, int, int, int>> pafPtrIntoVector(
        const Array<T>& pairScores, const T* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyPartPairs)
    {
        try
        {
            // Result is a std::vector<std::tuple<double, double, int, int, int>> with:
            // (totalScore, PAFscore, pairIndex, indexA, indexB)
            // totalScore is first to simplify later sorting
            std::vector<std::tuple<T, T, int, int, int>> pairConnections;

            // Get all PAF pairs in a single std::vector
            const auto peaksOffset = 3*(maxPeaks+1);
            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberPeaksA = intRound(candidateAPtr[0]);
                const auto numberPeaksB = intRound(candidateBPtr[0]);
                const auto firstIndex = (int)pairIndex*pairScores.getSize(1)*pairScores.getSize(2);
                // E.g., neck-nose connection. For each neck
                for (auto indexA = 0; indexA < numberPeaksA; indexA++)
                {
                    const auto iIndex = firstIndex + indexA*pairScores.getSize(2);
                    // E.g., neck-nose connection. For each nose
                    for (auto indexB = 0; indexB < numberPeaksB; indexB++)
                    {
                        const auto scoreAB = pairScores[iIndex + indexB];

                        // E.g., neck-nose connection. If possible PAF between neck indexA, nose indexB --> add
                        // parts score + connection score
                        if (scoreAB > 1e-6)
                        {
                            // totalScore - Only used for sorting
                            // // Original totalScore
                            // const auto totalScore = scoreAB;
                            // Improved totalScore
                            // Improved to avoid too much weight in the PAF between 2 elements, adding some weight
                            // on their confidence (avoid connecting high PAFs on very low-confident keypoints)
                            const auto indexScoreA = bodyPartA*peaksOffset + (indexA+1)*3 + 2;
                            const auto indexScoreB = bodyPartB*peaksOffset + (indexB+1)*3 + 2;
                            const auto totalScore = scoreAB
                                                  + T(0.1)*peaksPtr[indexScoreA]
                                                  + T(0.1)*peaksPtr[indexScoreB];
                            // +1 because peaksPtr starts with counter
                            pairConnections.emplace_back(
                                std::make_tuple(totalScore, scoreAB, pairIndex, indexA+1, indexB+1));
                        }
                    }
                }
            }

            // Sort rows in descending order based on its first element (`totalScore`)
            if (!pairConnections.empty())
                std::sort(pairConnections.begin(), pairConnections.end(),
                          std::greater<std::tuple<double, double, int, int, int>>());

            // Return result
            return pairConnections;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template <typename T>
    std::vector<std::pair<std::vector<int>, T>> pafVectorIntoPeopleVector(
        const std::vector<std::tuple<T, T, int, int, int>>& pairConnections, const T* const peaksPtr,
        const int maxPeaks, const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts)
    {
        try
        {
            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: person subset score
            std::vector<std::pair<std::vector<int>, T>> peopleVector;
            const auto vectorSize = numberBodyParts+1;
            const auto peaksOffset = (maxPeaks+1);
            // Save which body parts have been already assigned
            std::vector<int> personAssigned(numberBodyParts*maxPeaks, -1);
            // Iterate over each PAF pair connection detected
            // E.g., neck1-nose2, neck5-Lshoulder0, etc.
            for (const auto& pairConnection : pairConnections)
            {
                // Read pairConnection
                // // Total score - only required for previous sort
                // const auto totalScore = std::get<0>(pairConnection);
                const auto pafScore = std::get<1>(pairConnection);
                const auto pairIndex = std::get<2>(pairConnection);
                const auto indexA = std::get<3>(pairConnection);
                const auto indexB = std::get<4>(pairConnection);
                // Derived data
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];

                const auto indexScoreA = (bodyPartA*peaksOffset + indexA)*3 + 2;
                const auto indexScoreB = (bodyPartB*peaksOffset + indexB)*3 + 2;
                // -1 because indexA and indexB are 1-based
                auto& aAssigned = personAssigned[bodyPartA*maxPeaks+indexA-1];
                auto& bAssigned = personAssigned[bodyPartB*maxPeaks+indexB-1];
                // Debugging
                #ifdef DEBUG
                    if (indexA-1 > peaksOffset || indexA <= 0)
                        error("Something is wrong: " + std::to_string(indexA)
                              + " vs. " + std::to_string(peaksOffset) + ". Contact us.",
                              __LINE__, __FUNCTION__, __FILE__);
                    if (indexB-1 > peaksOffset || indexB <= 0)
                        error("Something is wrong: " + std::to_string(indexB)
                              + " vs. " + std::to_string(peaksOffset) + ". Contact us.",
                              __LINE__, __FUNCTION__, __FILE__);
                #endif

                // Different cases:
                //     1. A & B not assigned yet: Create new person
                //     2. A assigned but not B: Add B to person with A (if no another B there)
                //     3. B assigned but not A: Add A to person with B (if no another A there)
                //     4. A & B already assigned to same person (circular/redundant PAF): Update person score
                //     5. A & B already assigned to different people: Merge people if keypoint intersection is null
                // 1. A & B not assigned yet: Create new person
                if (aAssigned < 0 && bAssigned < 0)
                {
                    // Keypoint indexes
                    std::vector<int> rowVector(vectorSize, 0);
                    rowVector[bodyPartA] = indexScoreA;
                    rowVector[bodyPartB] = indexScoreB;
                    // Number keypoints
                    rowVector.back() = 2;
                    // Score
                    const auto personScore = peaksPtr[indexScoreA] + peaksPtr[indexScoreB] + pafScore;
                    // Set associated personAssigned as assigned
                    aAssigned = (int)peopleVector.size();
                    bAssigned = aAssigned;
                    // Create new personVector
                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                }
                // 2. A assigned but not B: Add B to person with A (if no another B there)
                // or
                // 3. B assigned but not A: Add A to person with B (if no another A there)
                else if ((aAssigned >= 0 && bAssigned < 0)
                    || (aAssigned < 0 && bAssigned >= 0))
                {
                    // Assign person1 to one where xAssigned >= 0
                    const auto assigned1 = (aAssigned >= 0 ? aAssigned : bAssigned);
                    auto& assigned2 = (aAssigned >= 0 ? bAssigned : aAssigned);
                    const auto bodyPart2 = (aAssigned >= 0 ? bodyPartB : bodyPartA);
                    const auto indexScore2 = (aAssigned >= 0 ? indexScoreB : indexScoreA);
                    // Person index
                    auto& personVector = peopleVector[assigned1];
                    // Debugging
                    #ifdef DEBUG
                        const auto bodyPart1 = (aAssigned >= 0 ? bodyPartA : bodyPartB);
                        const auto indexScore1 = (aAssigned >= 0 ? indexScoreA : indexScoreB);
                        const auto index1 = (aAssigned >= 0 ? indexA : indexB);
                        if ((unsigned int)personVector.first.at(bodyPart1) != indexScore1)
                            error("Something is wrong: "
                                  + std::to_string((personVector.first[bodyPart1]-2)/3-bodyPart1*peaksOffset)
                                  + " vs. " + std::to_string((indexScore1-2)/3-bodyPart1*peaksOffset) + " vs. "
                                  + std::to_string(index1) + ". Contact us.",
                                  __LINE__, __FUNCTION__, __FILE__);
                    #endif
                    // If person with 1 does not have a 2 yet
                    if (personVector.first[bodyPart2] == 0)
                    {
                        // Update keypoint indexes
                        personVector.first[bodyPart2] = indexScore2;
                        // Update number keypoints
                        personVector.first.back()++;
                        // Update score
                        personVector.second += peaksPtr[indexScore2] + pafScore;
                        // Set associated personAssigned as assigned
                        assigned2 = assigned1;
                    }
                    // Otherwise, ignore this B because the previous one came from a higher PAF-confident score
                }
                // 4. A & B already assigned to same person (circular/redundant PAF): Update person score
                else if (aAssigned >=0 && bAssigned >=0 && aAssigned == bAssigned)
                    peopleVector[aAssigned].second += pafScore;
                // 5. A & B already assigned to different people: Merge people if keypoint intersection is null
                // I.e., that the keypoints in person A and B do not overlap
                else if (aAssigned >=0 && bAssigned >=0 && aAssigned != bAssigned)
                {
                    // Assign person1 to the one with lowest index for 2 reasons:
                    //     1. Speed up: Removing an element from std::vector is cheaper for latest elements
                    //     2. Avoid harder index update: Updated elements in person1ssigned would depend on
                    //        whether person1 > person2 or not: element = aAssigned - (person2 > person1 ? 1 : 0)
                    const auto assigned1 = (aAssigned < bAssigned ? aAssigned : bAssigned);
                    const auto assigned2 = (aAssigned < bAssigned ? bAssigned : aAssigned);
                    auto& person1 = peopleVector[assigned1].first;
                    const auto& person2 = peopleVector[assigned2].first;
                    // Check if complementary
                    // Defining found keypoint indexes in personA as kA, and analogously kB
                    // Complementary if and only if kA intersection kB = empty. I.e., no common keypoints
                    bool complementary = true;
                    for (auto part = 0u ; part < numberBodyParts ; part++)
                    {
                        if (person1[part] > 0 && person2[part] > 0)
                        {
                            complementary = false;
                            break;
                        }
                    }
                    // If complementary, merge both people into 1
                    if (complementary)
                    {
                        // Update keypoint indexes
                        for (auto part = 0u ; part < numberBodyParts ; part++)
                            if (person1[part] == 0)
                                person1[part] = person2[part];
                        // Update number keypoints
                        person1.back() += person2.back();
                        // Update score
                        peopleVector[assigned1].second += peopleVector[assigned2].second + pafScore;
                        // Erase the non-merged person
                        peopleVector.erase(peopleVector.begin()+assigned2);
                        // Update associated personAssigned (person indexes have changed)
                        for (auto& element : personAssigned)
                        {
                            if (element == assigned2)
                                element = assigned1;
                            else if (element > assigned2)
                                element--;
                        }
                    }
                }
            }
            // Return result
            return peopleVector;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template <typename T>
    void removePeopleBelowThresholds(std::vector<int>& validSubsetIndexes, int& numberPeople,
                                     const std::vector<std::pair<std::vector<int>, T>>& peopleVector,
                                     const unsigned int numberBodyParts, const int minSubsetCnt,
                                     const T minSubsetScore, const int maxPeaks,
                                     const bool maximizePositives)
    {
        try
        {
            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            numberPeople = 0;
            validSubsetIndexes.clear();
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, peopleVector.size()));
            for (auto index = 0u ; index < peopleVector.size() ; index++)
            {
                auto personCounter = peopleVector[index].first.back();
                // Foot keypoints do not affect personCounter (too many false positives,
                // same foot usually appears as both left and right keypoints)
                // Pros: Removed tons of false positives
                // Cons: Standalone leg will never be recorded
                if (!maximizePositives && numberBodyParts == 25)
                {
                    // No consider foot keypoints for that
                    for (auto i = 19 ; i < 25 ; i++)
                        personCounter -= (peopleVector[index].first.at(i) > 0);
                }
                const auto personScore = peopleVector[index].second;
                if (personCounter >= minSubsetCnt && (personScore/personCounter) >= minSubsetScore)
                {
                    numberPeople++;
                    validSubsetIndexes.emplace_back(index);
                    if (numberPeople == maxPeaks)
                        break;
                }
                else if ((personCounter < 1 && numberBodyParts != 25) || personCounter < 0)
                    error("Bad personCounter (" + std::to_string(personCounter) + "). Bug in this"
                          " function if this happens.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void peopleVectorToPeopleArray(Array<T>& poseKeypoints, Array<T>& poseScores, const T scaleFactor,
                                   const std::vector<std::pair<std::vector<int>, T>>& peopleVector,
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
                const auto& personPair = peopleVector[validSubsetIndexes[person]];
                const auto& personVector = personPair.first;
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
                    const auto bodyPartIndex = personVector[bodyPart];
                    if (bodyPartIndex > 0)
                    {
                        poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex-2] * scaleFactor;
                        poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex-1] * scaleFactor;
                        poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
                    }
                }
                poseScores[person] = personPair.second / T(numberBodyPartsAndPAFs);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

//     template <typename T>
//     void connectDistanceStar(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
//                              const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
//                              const int maxPeaks, const T scaleFactor, const unsigned int numberBodyParts,
//                              const unsigned int bodyPartPairsSize)
//     {
//         try
//         {
//             // poseKeypoints from neck-part distances
//             if (poseModel == PoseModel::BODY_25D)
//             {
//                 const auto scaleDownFactor = 8;
//                 Array<T> poseKeypoints2 = poseKeypoints.clone();
//                 const auto rootIndex = 1;
//                 const auto rootNumberIndex = rootIndex*(maxPeaks+1)*3;
//                 const auto numberPeople = intRound(peaksPtr[rootNumberIndex]);
//                 poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3}, 0);
//                 poseScores.reset(numberPeople, 0);
//                 // // 48 channels
//                 // const std::vector<float> AVERAGE{
//                 //     0.f, -2.76364f, -1.3345f, 0.f,   -1.95322f, 3.95679f, -1.20664f, 4.76543f,
//                 //     1.3345f, 0.f, 1.92318f, 3.96891f,   1.17999f, 4.7901f, 0.f, 7.72201f,
//                 //     -0.795236f, 7.74017f, -0.723963f,   11.209f, -0.651316f, 15.6972f,
//                 //     0.764623f, 7.74869f, 0.70755f,   11.2307f, 0.612832f, 15.7281f,
//                 //     -0.123134f, -3.43515f,   0.111775f, -3.42761f,
//                 //     -0.387066f, -3.16603f,   0.384038f, -3.15951f,
//                 //     0.344764f, 12.9666f, 0.624157f,   12.9057f, 0.195454f, 12.565f,
//                 //     -1.06074f, 12.9951f, -1.2427f,   12.9309f, -0.800837f, 12.5845f};
//                 // const std::vector<float> SIGMA{
//                 //     3.39629f, 3.15605f, 3.16913f, 1.8234f,   5.82252f, 5.05674f, 7.09876f, 6.64574f,
//                 //     3.16913f, 1.8234f, 5.79415f, 5.01424f,   7.03866f, 6.62427f, 5.52593f, 6.75962f,
//                 //     5.91224f, 6.87241f, 8.66473f,   10.1792f, 11.5871f, 13.6565f,
//                 //     5.86653f, 6.89568f, 8.68067f,   10.2127f, 11.5954f, 13.6722f,
//                 //     3.3335f, 3.49128f,   3.34476f, 3.50079f,
//                 //     2.93982f, 3.11151f,   2.95006f, 3.11004f,
//                 //     9.69408f, 7.58921f, 9.71193f,   7.44185f, 9.19343f, 7.11157f,
//                 //     9.16848f, 7.86122f, 9.07613f,   7.83682f, 8.91951f, 7.33715f};
//                 // 50 channels
//                 const std::vector<float> AVERAGE{
//                     0.f, -6.55251f,
//                     0.f, -4.15062f, -1.48818f, -4.15506f,   -2.22408f, -0.312264f, -1.42204f, 0.588495f,
//                     1.51044f, -4.14629f, 2.2113f, -0.312283f,   1.41081f, 0.612377f, -0.f, 3.41112f,
//                     -0.932306f, 3.45504f, -0.899812f,   6.79837f, -0.794223f, 11.4972f,
//                     0.919047f, 3.46442, 0.902314f,   6.81245f, 0.79518f, 11.5132f,
//                     -0.243982f, -7.07925f,   0.28065f, -7.07398f,
//                     -0.792812f, -7.09374f,   0.810145f, -7.06958f,
//                     0.582387f, 7.46846f, 0.889349f,   7.40577f, 0.465088f, 7.03969f,
//                     -0.96686f, 7.46148f, -1.20773f,   7.38834f, -0.762135f, 6.99575f};
//                 const std::vector<float> SIGMA{
//                     7.26789f, 9.70751f,
//                     6.29588f, 8.93472f, 6.97401f, 9.13746f,   7.49632f, 9.44757f, 8.06695f, 9.97319f,
//                     6.99726f, 9.14608f, 7.50529f, 9.43568f,   8.05888f, 9.98207f, 6.38929f, 9.29314f,
//                     6.71801f, 9.39271f, 8.00608f,   10.6141f, 10.3416f, 12.7812f,
//                     6.69875f, 9.41407f, 8.01876f,   10.637f, 10.3475f, 12.7849f,
//                     7.30923f, 9.7324f,   7.27886f, 9.73406f,
//                     7.35978f, 9.7289f,   7.28914f, 9.67711f,
//                     7.93153f, 8.10845f, 7.95577f,   8.01729f, 7.56865f, 7.87314f,
//                     7.4655f, 8.25336f, 7.43958f,   8.26333f, 7.33667f, 7.97446f};
//                 // To get ideal distance
//                 const auto numberBodyPartsAndBkgAndPAFChannels = numberBodyParts + 1 + bodyPartPairsSize;
//                 const auto heatMapOffset = heatMapSize.area();
//                 // For each person
//                 for (auto p = 0 ; p < numberPeople ; p++)
//                 {
//                     // For root (neck) position
//                     // bpOrig == rootIndex
//                     const auto rootXYSIndex = rootNumberIndex+3*(1+p);
//                     // Set (x,y,score)
//                     const auto rootX = scaleFactor*peaksPtr[rootXYSIndex];
//                     const auto rootY = scaleFactor*peaksPtr[rootXYSIndex+1];
//                     poseKeypoints[{p,rootIndex,0}] = rootX;
//                     poseKeypoints[{p,rootIndex,1}] = rootY;
//                     poseKeypoints[{p,rootIndex,2}] = peaksPtr[rootXYSIndex+2];
//                     // For each body part
//                     for (auto bpOrig = 0 ; bpOrig < (int)numberBodyParts ; bpOrig++)
//                     {
//                         if (bpOrig != rootIndex)
//                         {
//                             // // 48 channels
//                             // const auto bpChannel = (bpOrig < rootIndex ? bpOrig : bpOrig-1);
//                             // 50 channels
//                             const auto bpChannel = bpOrig;
//                             // Get ideal distance
//                             const auto offsetIndex = numberBodyPartsAndBkgAndPAFChannels + 2*bpChannel;
//                             const auto* mapX = heatMapPtr + offsetIndex * heatMapOffset;
//                             const auto* mapY = heatMapPtr + (offsetIndex+1) * heatMapOffset;
//                             const auto increaseRatio = scaleFactor*scaleDownFactor;
//                             // Set (x,y) coordinates from the distance
//                             const auto indexChannel = 2*bpChannel;
//                             // // Not refined method
//                             // const auto index = intRound(
//                             //     rootY/scaleFactor)*heatMapSize.x + intRound(rootX/scaleFactor);
//                             // const Point<T> neckPartDist{
//                             //     increaseRatio*(mapX[index]*SIGMA[indexChannel]+AVERAGE[indexChannel]),
//                             //     increaseRatio*(mapY[index]*SIGMA[indexChannel+1]+AVERAGE[indexChannel+1])};
//                             // poseKeypoints[{p,bpOrig,0}] = rootX + neckPartDist.x;
//                             // poseKeypoints[{p,bpOrig,1}] = rootY + neckPartDist.y;
//                             // Refined method
//                             const auto constant = 5;
//                             Point<T> neckPartDistRefined{0, 0};
//                             auto counterRefinements = 0;
//                             // We must keep it inside the image size
//                             for (auto y = fastMax(0, intRound(rootY/scaleFactor) - constant);
//                                  y < fastMin(heatMapSize.y, intRound(rootY/scaleFactor) + constant+1) ; y++)
//                             {
//                                 for (auto x = fastMax(0, intRound(rootX/scaleFactor) - constant);
//                                      x < fastMin(heatMapSize.x, intRound(rootX/scaleFactor) + constant+1) ; x++)
//                                 {
//                                     const auto index = y*heatMapSize.x + x;
//                                     neckPartDistRefined.x += mapX[index];
//                                     neckPartDistRefined.y += mapY[index];
//                                     counterRefinements++;
//                                 }
//                             }
//                             neckPartDistRefined = Point<T>{
//                                 neckPartDistRefined.x*SIGMA[indexChannel]+counterRefinements*AVERAGE[indexChannel],
//                                 neckPartDistRefined.y*SIGMA[indexChannel+1]+counterRefinements*AVERAGE[indexChannel+1],
//                             };
//                             neckPartDistRefined *= increaseRatio/counterRefinements;
//                             const auto partX = rootX + neckPartDistRefined.x;
//                             const auto partY = rootY + neckPartDistRefined.y;
//                             poseKeypoints[{p,bpOrig,0}] = partX;
//                             poseKeypoints[{p,bpOrig,1}] = partY;
//                             // Set (temporary) body part score
//                             poseKeypoints[{p,bpOrig,2}] = T(0.0501);
//                             // Associate estimated keypoint with closest one
//                             const auto xCleaned = fastMax(0, fastMin(heatMapSize.x-1, intRound(partX/scaleFactor)));
//                             const auto yCleaned = fastMax(0, fastMin(heatMapSize.y-1, intRound(partY/scaleFactor)));
//                             const auto partConfidence = heatMapPtr[
//                                 bpOrig * heatMapOffset + yCleaned*heatMapSize.x + xCleaned];
//                             // If partConfidence is big enough, it means we are close to a keypoint
//                             if (partConfidence > T(0.05))
//                             {
//                                 const auto candidateNumberIndex = bpOrig*(maxPeaks+1)*3;
//                                 const auto numberCandidates = intRound(peaksPtr[candidateNumberIndex]);
//                                 int closestIndex = -1;
//                                 T closetValue = std::numeric_limits<T>::max();
//                                 for (auto i = 0 ; i < numberCandidates ; i++)
//                                 {
//                                     const auto candidateXYSIndex = candidateNumberIndex+3*(1+i);
//                                     const auto diffX = partX-scaleFactor*peaksPtr[candidateXYSIndex];
//                                     const auto diffY = partY-scaleFactor*peaksPtr[candidateXYSIndex+1];
//                                     const auto dist = (diffX*diffX + diffY*diffY);
//                                     if (closetValue > dist)
//                                     {
//                                         closetValue = dist;
//                                         closestIndex = candidateXYSIndex;
//                                     }
//                                 }
//                                 if (closestIndex != -1)
//                                 {
//                                     poseKeypoints[{p,bpOrig,0}] = scaleFactor*peaksPtr[closestIndex];
//                                     poseKeypoints[{p,bpOrig,1}] = scaleFactor*peaksPtr[closestIndex+1];
//                                     // Set body part score
//                                     poseKeypoints[{p,bpOrig,2}] = peaksPtr[closestIndex+2];
//                                 }
//                             }
//                             // Set poseScore
//                             poseScores[p] += poseKeypoints[{p,bpOrig,2}];
//                         }
//                     }
//                 }
//             }
//         }
//         catch (const std::exception& e)
//         {
//             error(e.what(), __LINE__, __FUNCTION__, __FILE__);
//         }
//     }

//     const std::vector<float> AVERAGE{
//         0.f, -6.55251f,
//         0.f, -4.15062f, -1.48818, -4.15506f,   -2.22408f, -0.312264f, -1.42204f, 0.588495f,
//         1.51044f, -4.14629f, 2.2113f, -0.312283f,   1.41081f, 0.612377f, -0.f, 3.41112f,
//         -0.932306f, 3.45504f, -0.899812f,   6.79837f, -0.794223f, 11.4972f,
//         0.919047f, 3.46442f, 0.902314f,   6.81245f, 0.79518f, 11.5132f,
//         -0.243982f, -7.07925f,   0.28065f, -7.07398f,
//         -0.792812f, -7.09374f,   0.810145f, -7.06958f,
//         0.582387f, 7.46846f, 0.889349f,   7.40577f, 0.465088f, 7.03969f,
//         -0.96686f, 7.46148f, -1.20773f,   7.38834f, -0.762135f, 6.99575f};
//     const std::vector<float> SIGMA{
//         7.26789f, 9.70751f,
//         6.29588f, 8.93472f, 6.97401f, 9.13746f,   7.49632f, 9.44757f, 8.06695f, 9.97319f,
//         6.99726f, 9.14608f, 7.50529f, 9.43568f,   8.05888f, 9.98207f, 6.38929f, 9.29314f,
//         6.71801f, 9.39271f, 8.00608f,   10.6141f, 10.3416f, 12.7812f,
//         6.69875f, 9.41407f, 8.01876f,   10.637f, 10.3475f, 12.7849f,
//         7.30923f, 9.7324f,   7.27886f, 9.73406f,
//         7.35978f, 9.7289f,   7.28914f, 9.67711f,
//         7.93153f, 8.10845f, 7.95577f,   8.01729f, 7.56865f, 7.87314f,
//         7.4655f, 8.25336f, 7.43958f,   8.26333f, 7.33667f, 7.97446f};
//     template <typename T>
//     std::array<T,3> regressPart(const Array<T>& person, const int rootIndex, const int targetIndex,
//                                 const int scaleDownFactor, const T* const heatMapPtr, const T* const peaksPtr,
//                                 const Point<int>& heatMapSize, const int maxPeaks, const T scaleFactor,
//                                 const unsigned int numberBodyPartsAndBkgAndPAFChannels)
//     {
//         try
//         {
//             std::array<T,3> result{0,0,0};
//             // poseKeypoints from neck-part distances
//             if (targetIndex != rootIndex && person[{rootIndex,2}] > T(0.05))
//             {
//                 // Set (x,y)
//                 const auto rootX = person[{rootIndex,0}];
//                 const auto rootY = person[{rootIndex,1}];
//                 // Get ideal distance
//                 const auto indexChannel = 2*targetIndex;
//                 const auto offsetIndex = numberBodyPartsAndBkgAndPAFChannels + indexChannel;
//                 const auto heatMapOffset = heatMapSize.area();
//                 const auto* mapX = heatMapPtr + offsetIndex * heatMapOffset;
//                 const auto* mapY = heatMapPtr + (offsetIndex+1) * heatMapOffset;
//                 const auto increaseRatio = scaleFactor*scaleDownFactor;
//                 // // Not refined method
//                 // const auto index = intRound(rootY/scaleFactor)*heatMapSize.x + intRound(rootX/scaleFactor);
//                 // const Point<T> neckPartDist{
//                 //     increaseRatio*(mapX[index]*SIGMA[indexChannel]+AVERAGE[indexChannel]),
//                 //     increaseRatio*(mapY[index]*SIGMA[indexChannel+1]+AVERAGE[indexChannel+1])};
//                 // poseKeypoints[{p,targetIndex,0}] = rootX + neckPartDist.x;
//                 // poseKeypoints[{p,targetIndex,1}] = rootY + neckPartDist.y;
//                 // Refined method
//                 const auto constant = 5;
//                 Point<T> neckPartDistRefined{0, 0};
//                 auto counterRefinements = 0;
//                 // We must keep it inside the image size
//                 for (auto y = fastMax(0, intRound(rootY/scaleFactor) - constant);
//                      y < fastMin(heatMapSize.y, intRound(rootY/scaleFactor) + constant+1) ; y++)
//                 {
//                     for (auto x = fastMax(0, intRound(rootX/scaleFactor) - constant);
//                          x < fastMin(heatMapSize.x, intRound(rootX/scaleFactor) + constant+1) ; x++)
//                     {
//                         const auto index = y*heatMapSize.x + x;
//                         neckPartDistRefined.x += mapX[index];
//                         neckPartDistRefined.y += mapY[index];
//                         counterRefinements++;
//                     }
//                 }
//                 neckPartDistRefined = Point<T>{
//                     neckPartDistRefined.x*SIGMA[indexChannel]+counterRefinements*AVERAGE[indexChannel],
//                     neckPartDistRefined.y*SIGMA[indexChannel+1]+counterRefinements*AVERAGE[indexChannel+1],
//                 };
//                 neckPartDistRefined *= increaseRatio/counterRefinements;
//                 const auto partX = rootX + neckPartDistRefined.x;
//                 const auto partY = rootY + neckPartDistRefined.y;
//                 result[0] = partX;
//                 result[1] = partY;
//                 // Set (temporary) body part score
//                 result[2] = T(0.0501);
//                 // Associate estimated keypoint with closest one
//                 const auto xCleaned = fastMax(0, fastMin(heatMapSize.x-1, intRound(partX/scaleFactor)));
//                 const auto yCleaned = fastMax(0, fastMin(heatMapSize.y-1, intRound(partY/scaleFactor)));
//                 const auto partConfidence = heatMapPtr[
//                     targetIndex * heatMapOffset + yCleaned*heatMapSize.x + xCleaned];
//                 // If partConfidence is big enough, it means we are close to a keypoint
//                 if (partConfidence > T(0.05))
//                 {
//                     const auto candidateNumberIndex = targetIndex*(maxPeaks+1)*3;
//                     const auto numberCandidates = intRound(peaksPtr[candidateNumberIndex]);
//                     int closestIndex = -1;
//                     T closetValue = std::numeric_limits<T>::max();
//                     for (auto i = 0 ; i < numberCandidates ; i++)
//                     {
//                         const auto candidateXYSIndex = candidateNumberIndex+3*(1+i);
//                         const auto diffX = partX-scaleFactor*peaksPtr[candidateXYSIndex];
//                         const auto diffY = partY-scaleFactor*peaksPtr[candidateXYSIndex+1];
//                         const auto dist = (diffX*diffX + diffY*diffY);
//                         if (closetValue > dist)
//                         {
//                             closetValue = dist;
//                             closestIndex = candidateXYSIndex;
//                         }
//                     }
//                     if (closestIndex != -1)
//                     {
//                         result[0] = scaleFactor*peaksPtr[closestIndex];
//                         result[1] = scaleFactor*peaksPtr[closestIndex+1];
//                         // Set body part score
//                         result[2] = peaksPtr[closestIndex+2];
//                     }
//                 }
//             }
//             return result;
//         }
//         catch (const std::exception& e)
//         {
//             error(e.what(), __LINE__, __FUNCTION__, __FILE__);
//             return std::array<T,3>{};
//         }
//     }

//     template <typename T>
//     void connectDistanceMultiStar(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
//                                   const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
//                                   const int maxPeaks, const T scaleFactor, const unsigned int numberBodyParts,
//                                   const unsigned int bodyPartPairsSize)
//     {
//         try
//         {
//             // poseKeypoints from neck-part distances
//             if (poseModel == PoseModel::BODY_25D)
//             {
//                 // Add all the root elements (necks)
//                 const std::vector<int> keypointsSize = {(int)numberBodyParts, 3};
//                 // Initial #people = number root elements
//                 const auto rootIndex = 1;
//                 std::vector<Array<T>> poseKeypointsTemp;
//                 // Iterate for each body part
//                 const std::array<int, 25> MAPPING{
//                     1, 8, 0, 2,5,9,12, 3,6,10,13, 15,16, 4,7,11,14, 17,18, 19,22,20,23,21,24};
//                 const auto numberBodyPartsAndBkgAndPAFChannels = numberBodyParts + 1 + bodyPartPairsSize;
//                 const auto scaleDownFactor = 8;
//                 for (auto index = 0u ; index < numberBodyParts ; index++)
//                 {
//                     const auto targetIndex = MAPPING[index];
//                     // Get all candidate keypoints
//                     const auto partNumberIndex = targetIndex*(maxPeaks+1)*3;
//                     const auto numberPartParts = intRound(peaksPtr[partNumberIndex]);
//                     std::vector<std::array<T, 3>> currentPartCandidates(numberPartParts);
//                     for (auto i = 0u ; i < currentPartCandidates.size() ; i++)
//                     {
//                         const auto baseIndex = partNumberIndex+3*(i+1);
//                         currentPartCandidates[i][0] = scaleFactor*peaksPtr[baseIndex];
//                         currentPartCandidates[i][1] = scaleFactor*peaksPtr[baseIndex+1];
//                         currentPartCandidates[i][2] = peaksPtr[baseIndex+2];
//                     }
//                     // Detect new body part for existing people
//                     // For each temporary person --> Add new targetIndex part
//                     for (auto& person : poseKeypointsTemp)
//                     {
//                         // Estimate new body part w.r.t. each already-detected body part
//                         for (auto rootMapIndex = 0u ; rootMapIndex < index ; rootMapIndex++)
//                         {
//                             const auto rootIndex = MAPPING[rootMapIndex];
//                             if (person[{rootIndex,2}] > T(0.0501))
//                             {
//                                 const auto result = regressPart(
//                                     person, rootIndex, targetIndex, scaleDownFactor, heatMapPtr, peaksPtr,
//                                     heatMapSize, maxPeaks, scaleFactor, numberBodyPartsAndBkgAndPAFChannels);
//                                 if (person[{targetIndex,2}] < result[2])
//                                 {
//                                     person[{targetIndex,0}] = result[0];
//                                     person[{targetIndex,1}] = result[1];
//                                     person[{targetIndex,2}] = result[2];
//                                 }
//                             }
//                         }
//                     }
//                     // Add leftovers body parts as new people
// if (targetIndex == rootIndex)
// {
//                     const auto currentSize = poseKeypointsTemp.size();
//                     poseKeypointsTemp.resize(currentSize+currentPartCandidates.size());
//                     for (auto p = 0u ; p < currentPartCandidates.size() ; p++)
//                     {
//                         poseKeypointsTemp[currentSize+p] = Array<T>(keypointsSize, 0.f);
//                         const auto baseIndex = 3*targetIndex;
//                         poseKeypointsTemp[currentSize+p][baseIndex  ] = currentPartCandidates[p][0];
//                         poseKeypointsTemp[currentSize+p][baseIndex+1] = currentPartCandidates[p][1];
//                         poseKeypointsTemp[currentSize+p][baseIndex+2] = currentPartCandidates[p][2];
//                     }

// }
//                 }
//                 // poseKeypoints: Reformat poseKeypointsTemp as poseKeypoints
//                 poseKeypoints.reset({(int)poseKeypointsTemp.size(), (int)numberBodyParts, 3}, 0);
//                 poseScores.reset(poseKeypoints.getSize(0), 0.f);
//                 const auto keypointArea = poseKeypoints.getSize(1)*poseKeypoints.getSize(2);
//                 for (auto p = 0 ; p < poseKeypoints.getSize(0) ; p++)
//                 {
//                     const auto pIndex = p*keypointArea;
//                     for (auto part = 0 ; part < poseKeypoints.getSize(1) ; part++)
//                     {
//                         const auto baseIndexTemp = 3*part;
//                         const auto baseIndex = pIndex+baseIndexTemp;
//                         poseKeypoints[baseIndex  ] = poseKeypointsTemp[p][baseIndexTemp];
//                         poseKeypoints[baseIndex+1] = poseKeypointsTemp[p][baseIndexTemp+1];
//                         poseKeypoints[baseIndex+2] = poseKeypointsTemp[p][baseIndexTemp+2];
//                         // Set poseScore
//                         poseScores[p] += poseKeypoints[baseIndex+2];
//                     }
//                 }
//             }
//         }
//         catch (const std::exception& e)
//         {
//             error(e.what(), __LINE__, __FUNCTION__, __FILE__);
//         }
//     }

    template <typename T>
    void connectBodyPartsCpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const bool maximizePositives)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = (unsigned int)(bodyPartPairs.size() / 2);
            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: person subset score
            const auto peopleVector = createPeopleVector(
                heatMapPtr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
                bodyPartPairs, numberBodyParts, numberBodyPartPairs);

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

            // Experimental code
            if (poseModel == PoseModel::BODY_25D)
                error("BODY_25D is an experimental branch which is not usable.", __LINE__, __FUNCTION__, __FILE__);
//                 connectDistanceMultiStar(poseKeypoints, poseScores, heatMapPtr, peaksPtr, poseModel, heatMapSize,
//                                          maxPeaks, scaleFactor, numberBodyParts, bodyPartPairs.size());
//                 connectDistanceStar(poseKeypoints, poseScores, heatMapPtr, peaksPtr, poseModel, heatMapSize,
//                                     maxPeaks, scaleFactor, numberBodyParts, bodyPartPairs.size());
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template OP_API void connectBodyPartsCpu(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float* const heatMapPtr,
        const float* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const float interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
        const float minSubsetScore, const float scaleFactor, const bool maximizePositives);
    template OP_API void connectBodyPartsCpu(
        Array<double>& poseKeypoints, Array<double>& poseScores, const double* const heatMapPtr,
        const double* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const double interMinAboveThreshold, const double interThreshold, const int minSubsetCnt,
        const double minSubsetScore, const double scaleFactor, const bool maximizePositives);

    template OP_API std::vector<std::pair<std::vector<int>, float>> createPeopleVector(
        const float* const heatMapPtr, const float* const peaksPtr, const PoseModel poseModel,
        const Point<int>& heatMapSize, const int maxPeaks, const float interThreshold,
        const float interMinAboveThreshold, const std::vector<unsigned int>& bodyPartPairs,
        const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
        const Array<float>& precomputedPAFs);
    template OP_API std::vector<std::pair<std::vector<int>, double>> createPeopleVector(
        const double* const heatMapPtr, const double* const peaksPtr, const PoseModel poseModel,
        const Point<int>& heatMapSize, const int maxPeaks, const double interThreshold,
        const double interMinAboveThreshold, const std::vector<unsigned int>& bodyPartPairs,
        const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
        const Array<double>& precomputedPAFs);

    template OP_API void removePeopleBelowThresholds(
        std::vector<int>& validSubsetIndexes, int& numberPeople,
        const std::vector<std::pair<std::vector<int>, float>>& peopleVector,
        const unsigned int numberBodyParts,
        const int minSubsetCnt, const float minSubsetScore, const int maxPeaks, const bool maximizePositives);
    template OP_API void removePeopleBelowThresholds(
        std::vector<int>& validSubsetIndexes, int& numberPeople,
        const std::vector<std::pair<std::vector<int>, double>>& peopleVector,
        const unsigned int numberBodyParts,
        const int minSubsetCnt, const double minSubsetScore, const int maxPeaks, const bool maximizePositives);

    template OP_API void peopleVectorToPeopleArray(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float scaleFactor,
        const std::vector<std::pair<std::vector<int>, float>>& peopleVector,
        const std::vector<int>& validSubsetIndexes, const float* const peaksPtr,
        const int numberPeople, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs);
    template OP_API void peopleVectorToPeopleArray(
        Array<double>& poseKeypoints, Array<double>& poseScores, const double scaleFactor,
        const std::vector<std::pair<std::vector<int>, double>>& peopleVector,
        const std::vector<int>& validSubsetIndexes, const double* const peaksPtr,
        const int numberPeople, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs);

    template OP_API std::vector<std::tuple<float, float, int, int, int>> pafPtrIntoVector(
        const Array<float>& pairScores, const float* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyPartPairs);
    template OP_API std::vector<std::tuple<double, double, int, int, int>> pafPtrIntoVector(
        const Array<double>& pairScores, const double* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyPartPairs);

    template OP_API std::vector<std::pair<std::vector<int>, float>> pafVectorIntoPeopleVector(
        const std::vector<std::tuple<float, float, int, int, int>>& pairConnections,
        const float* const peaksPtr, const int maxPeaks, const std::vector<unsigned int>& bodyPartPairs,
        const unsigned int numberBodyParts);
    template OP_API std::vector<std::pair<std::vector<int>, double>> pafVectorIntoPeopleVector(
        const std::vector<std::tuple<double, double, int, int, int>>& pairConnections,
        const double* const peaksPtr, const int maxPeaks, const std::vector<unsigned int>& bodyPartPairs,
        const unsigned int numberBodyParts);
}
