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
**Operating system** (`lsb_release -a` in Ubuntu):
**Installation mode**: CMake, sh script, manual Makefile installation, ... (Ubuntu); VS2015, VS2017, CMake, ... (Windows); ...?
**CUDA version** (`cat /usr/local/cuda/version.txt` in most cases):
**cuDNN version**:
**CMake version** (`cmake --version` in Ubuntu):
**Release or Debug mode**? (by defualt: release):
**3-D Reconstruction module added**? (by default: no):
**GPU model** (`nvidia-smi` in Ubuntu):
**Caffe version**: Default from OpenPose, custom version, ...?
**OpenCV version**: pre-compiled `apt-get install libopencv-dev` (only Ubuntu); OpenPose default (only Windows); compiled from source? If so, 2.4.9, 2.4.12, 3.1, 3.2?; ...?
Compiler (`gcc --version` in Ubuntu):
