OpenPose C++ API - How to Debug OpenPose
======================================================

# Finding Segmentation Faults
This is the faster method to debug a segmentation fault problem. Usual scenario: You are editing OpenPose source code and suddenly OpenPose returns segmentation fault when executed. In order to find where it occurs:
    1. Select one of the 2 options:
        1. Switch to debug mode.
        2. Go to `openpose/utilities/errorAndLog.hpp` and modify `dLog`:
            1. Comment `#ifndef NDEBUG` and its else and endif.
    2. Call OpenPose with `--logging_level 0 --disable_multi_thread`.
    3. At this point you have an idea of in which file class the segmentation fault is coming from. Now you can further isolate the error by iteratively adding the following line all over the code until you find the exact position of the segmentation fault: `log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);`
    4. After you have found the segmentation fault, remember to remove all the extra `log()` calls that you temporaryly added.
