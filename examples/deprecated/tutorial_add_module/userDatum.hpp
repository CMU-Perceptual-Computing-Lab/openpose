#ifndef OPENPOSE_TUTORIAL_USER_DATUM_HPP
#define OPENPOSE_TUTORIAL_USER_DATUM_HPP

#include <openpose/core/datum.hpp>
#include <openpose/core/common.hpp>

namespace op
{
    // If the user needs his own variables to be shared across Worker classes, he can directly add them here.
    struct UserDatum : public Datum
    {
        /**
         * Description of your variable.
         */
        bool boolThatUserNeedsForSomeReason;

        UserDatum() {};

        virtual ~UserDatum(){};
    };
}

#endif // OPENPOSE_TUTORIAL_USER_DATUM_HPP
