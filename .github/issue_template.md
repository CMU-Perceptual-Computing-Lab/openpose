### Posting rules
1. **Fill** the **Your System Configuration section (all of it!)** if you have some kind of error or performance question.
2. **No** questions about **training**. OpenPose only implements testing.
3. **No** questions about **Caffe installation errors/issues**. Check [Caffe](http://caffe.berkeleyvision.org) documentation and help for those errors.
4. **No** questions about **CUDA check failed errors**. These errors are usually fixed by re-installing CUDA, re-installing the proper cuDNN version, and re-compiling (or re-installing) OpenPose. Otherwise, check for help in CUDA forums.
5. **No duplicated** posts.
6. **No** posts about **questions already answered / clearly explained in** the **documentation** (e.g. **no more low-speed nor out-of-memory questions**).
7. Set a **proper issue title**: add the Ubuntu/Windows word and be specific (e.g. do not simple call it: `Compile error`).
8. Only English comments.
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
1. **General configuration**:
    - **Installation mode**: CMake, sh script, manual Makefile installation, ... (Ubuntu); CMake, ... (Windows); ...?
    - **Operating system** (`lsb_release -a` in Ubuntu):
    - **Release or Debug mode**? (by defualt: release):
    - Compiler (`gcc --version` in Ubuntu or VS version in Windows): 5.4.0, ... (Ubuntu); VS2015 Enterprise Update 3, VS2017 community, ... (Windows); ...?

2. **Non-default settings**:
    - **3-D Reconstruction module added**? (by default: no):
    - Any other custom CMake configuration with respect to the default version? (by default: no):

3. **3rd-party software**:
    - **Caffe version**: Default from OpenPose, custom version, ...?
    - **CMake version** (`cmake --version` in Ubuntu):
    - **OpenCV version**: pre-compiled `apt-get install libopencv-dev` (only Ubuntu); OpenPose default (only Windows); compiled from source? If so, 2.4.9, 2.4.12, 3.1, 3.2?; ...?

4. If **GPU mode** issue:
    - **CUDA version** (`cat /usr/local/cuda/version.txt` in most cases):
    - **cuDNN version**:
    - **GPU model** (`nvidia-smi` in Ubuntu):

5. If **CPU-only mode** issue:
    - **CPU model**:
    - Total **RAM memory** available:

6. If **speed performance** issue:
    - Report OpenPose timing speed based on [this link](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#profiling-speed).
