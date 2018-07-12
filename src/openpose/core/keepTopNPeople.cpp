#include <openpose/utilities/keypoint.hpp>
#include <openpose/core/keepTopNPeople.hpp>

namespace op
{
    KeepTopNPeople::KeepTopNPeople(const int numberPeopleMax) :
        mNumberPeopleMax{numberPeopleMax}
    {
    }

    Array<float> KeepTopNPeople::keepTopPeople(const Array<float>& peopleArray, const Array<float>& poseScores) const
    {
        try
        {
            // Remove people if #people > mNumberPeopleMax
            if (peopleArray.getVolume() > (unsigned int)mNumberPeopleMax && mNumberPeopleMax > 0)
            {
                // Security checks
                if (poseScores.getVolume() != (unsigned int) poseScores.getSize(0))
                    error("The poseFinalScores variable should be a Nx1 vector, not a multidimensional array.",
                          __LINE__, __FUNCTION__, __FILE__);
                if (peopleArray.getNumberDimensions() != 3)
                    error("The peopleArray variable should be a 3 dimensional array.",
                          __LINE__, __FUNCTION__, __FILE__);

                // Get poseFinalScores
                auto poseFinalScores = poseScores.clone();
                for (auto person = 0 ; person < (int)poseFinalScores.getVolume() ; person ++)
                    poseFinalScores[person] *= std::sqrt(getKeypointsArea(peopleArray, person, 0.05f));

                // Get threshold
                auto poseScoresSorted = poseFinalScores.clone();
                std::sort(poseScoresSorted.getPtr(), poseScoresSorted.getPtr() + poseScoresSorted.getSize(0),
                          std::greater<float>());
                const auto threshold = poseScoresSorted[mNumberPeopleMax-1];

                // Get number people above threshold
                auto numberPeopleAboveThreshold = 0;
                for (auto person = 0 ; person < (int)poseFinalScores.getVolume() ; person ++)
                    if (poseFinalScores[person] > threshold)
                        numberPeopleAboveThreshold++;

                // Remove extra people - Fille topPeopleArray
                // assignedPeopleOnThreshold avoids that people with repeated threshold remove higher elements. 
                // In our case, it will keep the first N people with score = threshold, while keeping all the
                // people with higher scores.
                // E.g., poseFinalScores = [0, 0.5, 0.5, 0.5, 1.0]; mNumberPeopleMax = 2
                // Naively, we could accidentally keep the first 2x 0.5 and remove the 1.0 threshold.
                // Our method keeps the first 0.5 and 1.0.
                Array<float> topPeopleArray({mNumberPeopleMax, peopleArray.getSize(1), peopleArray.getSize(2)});
                const auto personArea = peopleArray.getSize(1) * peopleArray.getSize(2);
                auto assignedPeopleOnThreshold = 0;
                auto nextPersonIndex = 0;
                const auto numberPeopleOnThresholdToBeAdded = mNumberPeopleMax - numberPeopleAboveThreshold;
                for (auto person = 0 ; person < (int)poseFinalScores.getVolume() ; person++)
                {
                    if (poseFinalScores[person] >= threshold)
                    {
                        // Check we don't copy 2 values in the threshold
                        if (poseFinalScores[person] == threshold)
                            assignedPeopleOnThreshold++;

                        // Copy person into people array
                        if (poseFinalScores[person] > threshold
                            || assignedPeopleOnThreshold <= numberPeopleOnThresholdToBeAdded)
                        {
                            const auto peopleArrayIndex = personArea*person;
                            const auto topArrayIndex = personArea*nextPersonIndex++;
                            std::copy(&peopleArray[peopleArrayIndex], &peopleArray[peopleArrayIndex]+personArea,
                                      &topPeopleArray[topArrayIndex]);
                        }
                    }
                }

                return topPeopleArray;
            }
            // If no changes required
            else
                return peopleArray;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }
}
