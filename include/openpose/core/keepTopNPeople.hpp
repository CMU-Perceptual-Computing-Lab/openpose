#ifndef OPENPOSE_CORE_KEEP_TOP_N_PEOPLE_HPP
#define OPENPOSE_CORE_KEEP_TOP_N_PEOPLE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API KeepTopNPeople
    {
    public:
        explicit KeepTopNPeople(const int numberPeopleMax);

        virtual ~KeepTopNPeople();

        Array<float> keepTopPeople(const Array<float>& peopleArrays, const Array<float>& poseScores) const;

    private:
        const int mNumberPeopleMax;
    };
}

#endif // OPENPOSE_CORE_KEEP_TOP_N_PEOPLE_HPP
