OpenPose C++ API - How to Develop OpenPose
======================================================





## OpenPose Coding Style
1. Naming:
    1. Class parameters should start with `m`, class pointers with `p`, shared_ptrs with `sp`, unique_ptrs with `up`, static parameters with `s`.
    2. Function and class parameters coding style is the same other than the previous point.
    3. Any parameters should not contain special characters, simply letters and numbers (preferred only letters) separated with upper case. E.g., `mThisIsAParameter`, `thisIsAParameter`.
    4. In addition, the names should be self-explanatory and not abbreviated. Good examples: `counter`, `thisIs. Bad examples: `ctr`, `var`.
2. Lenght:
    1. Lines should contain up to 120 characters.
3. Comments:
    1. Only `//` comments are allowed in the code, `/* */` should not be used.
    2. There should be a (at least) 1-line comment for each block of code inside each function.
4. Loops and statements:
    1. There should be a space between the keyword (`if`, `for`, etc) and the parenthesis, e.g.: `if (true)`. Wrong: `if(true)`. Note: So they can be easily located with Ctrl + F.
    2. Braces should be added in the following line with respect to the loop/statement keyword. See example in point 3.
    3. 1-line loops/statements should not contain braces. E.g.:
```
if (booleanParameter)
    anotherParameter = 25;
else
{
    anotherParameter = 2;
    differentParameter = 3;
}
```

5. Includes:
    1. They should be sorted in this order:
        1. Std libraries.
        2. OS libraries.
        3. 3rd party libraries (e.g. Caffe, OpenCV).
        4. OpenPose libraries.
        5. If it is a cpp file, the last one should be its own hpp.
    2. Inside each of the previous groups, it should be sorted alphabetically.
6. Functions arguments:
    1. It should first include the variables to be edited, and secondtly the const variables.
    2. Any variable that is not gonna be modified must be added with `const`.
7. Pointers:
    1. Pointers must be avoided if possible.
    2. If a pointer must be used, std::unique_ptr must be always be used.
    3. If the pointer must be shared, then std::shared_ptr.
    4. No `delete` keyword is allowed in OpenPose.





## Debugging C++ Code
### Finding Segmentation Faults
This is the faster method to debug a segmentation fault problem. Usual scenario: You are editing OpenPose source code and suddenly OpenPose returns segmentation fault when executed. In order to find where it occurs:

    1. Select one of the 2 options:
        1. Switch to debug mode.
        2. Go to `openpose/utilities/errorAndLog.hpp` and modify `dLog`:
            1. Comment `#ifndef NDEBUG` and its else and endif.
    2. Call OpenPose with `--logging_level 0 --disable_multi_thread`.
    3. At this point you have an idea of in which file class the segmentation fault is coming from. Now you can further isolate the error by iteratively adding the following line all over the code until you find the exact position of the segmentation fault: `log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);`
    4. After you have found the segmentation fault, remember to remove all the extra `log()` calls that you temporaryly added.





## Speed
### Measuring Runtime Speed
1. Enable `PROFILER_ENABLED` with CMake or in the `Makefile.config` file.
2. By default, it should print out average runtime info after 1000 frames. You can change this number with `--profile_speed`, e.g., `--profile_speed 100`.





## Accuracy
### Checking OpenPose Accuracy Quantitatively
1. Download OpenPose training code: https://github.com/CMU-Perceptual-Computing-Lab/openpose_train
2. Download val2017 set from COCO: http://images.cocodataset.org/zips/val2017.zip
3. Get JSONs in OpenPose: examples/tests/pose_accuracy_coco_val.sh
4. Get accuracy (Matlab): validation/f_getValidations.m

### Checking Ground-Truth Labes
From the [COCO dataset](http://cocodataset.org/#download):
1. Download 2014 or 2017 Train/Val annotations.
2. Download the [COCO API](https://github.com/cocodataset/cocoapi).
3. With the COCO API (either Python, Matlab, or LUA ones), you can check any image with the image ID (equivalent to the number in the image name).
