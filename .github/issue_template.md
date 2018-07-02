### Posting rules
1. **Fill** the **Your System Configuration section (all of it!)** if you have some kind of error or performance question.
2. **No** questions about **training**. OpenPose only implements testing.
3. **No** questions about **3rd party libraries**.
    - Caffe errors/issues, check [Caffe](http://caffe.berkeleyvision.org) documentation.
    - CUDA check failed errors: They are usually fixed by re-installing CUDA, then re-installing the proper cuDNN version, and then re-compiling (or re-installing) OpenPose. Otherwise, check for help in CUDA forums.
    - OpenCV errors: Install the default/pre-compiled OpenCV or check for online help.
4. **No duplicated** posts.
5. **No** posts about **questions already answered / clearly explained in** the **documentation** (e.g. **no more low-speed nor out-of-memory questions**).
6. Set a **proper issue title**: add the Ubuntu/Windows word and be specific (e.g. do not simple call it: `Compile error`).
7. Only English comments.
Issues/comments which do not follow these rules will be **ignored or removed** with no further clarification.



### Issue Summary



### Executed Command (if any)
Note: add `--logging_level 0 --disable_multi_thread` to get higher debug information.



### OpenPose Output (if any)



### Type of Issue
You might select multiple topics, delete the rest:
- Compilation/installation error
- Execution error
- Help wanted
- Question
- Enhancement / offering possible extensions / pull request / etc
- Other (type your own type)



### Your System Configuration
1. **OpenPose version**: Latest GitHub code? Or specific commit (e.g., d52878f)? Or specific version from `Release` section (e.g., 1.2.0)?

2. **General configuration**:
    - **Installation mode**: CMake, sh script, manual Makefile installation, ... (Ubuntu); CMake, ... (Windows); ...?
    - **Operating system** (`lsb_release -a` in Ubuntu):
    - **Release or Debug mode**? (by defualt: release):
    - Compiler (`gcc --version` in Ubuntu or VS version in Windows): 5.4.0, ... (Ubuntu); VS2015 Enterprise Update 3, VS2017 community, ... (Windows); ...?

3. **Non-default settings**:
    - **3-D Reconstruction module added**? (by default: no):
    - Any other custom CMake configuration with respect to the default version? (by default: no):

4. **3rd-party software**:
    - **Caffe version**: Default from OpenPose, custom version, ...?
    - **CMake version** (`cmake --version` in Ubuntu):
    - **OpenCV version**: pre-compiled `apt-get install libopencv-dev` (only Ubuntu); OpenPose default (only Windows); compiled from source? If so, 2.4.9, 2.4.12, 3.1, 3.2?; ...?

5. If **GPU mode** issue:
    - **CUDA version** (`cat /usr/local/cuda/version.txt` in most cases):
    - **cuDNN version**:
    - **GPU model** (`nvidia-smi` in Ubuntu):

6. If **CPU-only mode** issue:
    - **CPU brand & model**:
    - Total **RAM memory** available:

7. If **Windows** system:
    - Portable demo or compiled library?

8. If **speed performance** issue:
    - Report OpenPose timing speed based on [this link](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#profiling-speed).
