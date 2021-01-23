### Posting rules
1. **No duplicated posts, only 1 new post opened a day, and up to 2 opened a week**. Otherwise, extrict user bans will occur.
    - Check the [FAQ](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/05_faq.md) section, other GitHub issues, and general documentation before posting. E.g., **low-speed, out-of-memory, output format, 0-people detected, installation issues, ...**).
    - Keep posting all your issues in the same post.
    - No bans if you are unsure whether the post is duplicated!
2. **Fill all** the **Your System Configuration section** if you are facing an error or unexpected behavior. Some posts (e.g., feature requests) might not require it.
3. **No questions about training or 3rd party libraries**:
    - OpenPose only implements testing. For training, check [OpenPose train](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train).
    - Caffe errors/issues: Check [Caffe](http://caffe.berkeleyvision.org) documentation.
    - CUDA/cuDNN check failed errors: They are usually fixed by re-installing CUDA, then re-installing the proper cuDNN version, then rebooting, and then re-installing OpenPose. Otherwise, check Nvidia/CUDA/cuDNN forums.
    - OpenCV errors: Install the default/pre-compiled OpenCV or check for OpenCV online help.
4. Set a **proper issue title**: Add the OS (Ubuntu, Windows) and be specific (e.g., do not call it: `Error`).
5. Only English comments.
6. Remove these posting rules from your post but follow them!
Posts which do not follow these rules will be **ignored/deleted** and those **users banned** with no further clarification.



### Issue Summary



### Executed Command (if any)
Note: add `--logging_level 0 --disable_multi_thread` to get higher debug information.



### OpenPose Output (if any)



### Errors (if any)



### Type of Issue
Select the topic(s) on your post, delete the rest:
- Compilation/installation error
- Execution error
- Help wanted
- Question
- Enhancement / offering possible extensions / pull request / etc
- Other (type your own type)



### Your System Configuration
1. **Whole console output** (if errors appeared), paste the error to [PasteBin](https://pastebin.com/) and then paste the link here: LINK

2. **OpenPose version**: Latest GitHub code? Or specific commit (e.g., d52878f)? Or specific version from `Release` section (e.g., 1.2.0)?

3. **General configuration**:
    - **Installation mode**: CMake, sh script, manual Makefile installation, ... (Ubuntu); CMake, ... (Windows); ...?
    - **Operating system** (`lsb_release -a` in Ubuntu):
    - **Operating system version** (e.g., Ubuntu 16, Windows 10, ...):
    - **Release or Debug mode**? (by default: release):
    - Compiler (`gcc --version` in Ubuntu or VS version in Windows): 5.4.0, ... (Ubuntu); VS2015 Enterprise Update 3, VS2017 community, ... (Windows); ...?

4. **Non-default settings**:
    - **3-D Reconstruction module added**? (by default: no):
    - Any other custom CMake configuration with respect to the default version? (by default: no):

5. **3rd-party software**:
    - **Caffe version**: Default from OpenPose, custom version, ...?
    - **CMake version** (`cmake --version` in Ubuntu):
    - **OpenCV version**: pre-compiled `apt-get install libopencv-dev` (only Ubuntu); OpenPose default (only Windows); compiled from source? If so, 2.4.9, 2.4.12, 3.1, 3.2?; ...?

6. If **GPU mode** issue:
    - **CUDA version** (`cat /usr/local/cuda/version.txt` in most cases):
    - **cuDNN version**:
    - **GPU model** (`nvidia-smi` in Ubuntu):

7. If **CPU-only mode** issue:
    - **CPU brand & model**:
    - Total **RAM memory** available:

8. If **Python** API:
    - **Python version**: 2.7, 3.7, ...?
    - **Numpy version** (`python -c "import numpy; print numpy.version.version"` in Ubuntu):

9. If **Windows** system:
    - Portable demo or compiled library?

10. If **speed performance** issue:
    - Report OpenPose timing speed based on the [profiling documentation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/06_maximizing_openpose_speed.md#profiling-speed).
