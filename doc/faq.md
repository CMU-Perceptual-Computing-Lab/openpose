OpenPose - Frequently Asked Question (FAQ)
============================================

## Contents
1. [FAQ](#faq)
    1. [Errors](#errors)
        1. [Download Server Down](#download-server-down)
        2. [Out of Memory Error](#out-of-memory-error)
        3. [Video/Webcam Not Working](#videowebcam-not-working)
        4. [Cannot Find OpenPose.dll Error](#cannot-find-openposedll-error-windows)
        5. [Free Invalid Pointer Error](#free-invalid-pointer-error)
        6. [Source Directory does not Contain CMakeLists.txt (Windows)](#source-directory-does-not-contain-cmakeliststxt-windows)
        7. [Always Zero People Detected](#always-zero-people-detected)
        8. [Very Few People Detected](#very-few-people-detected)
        9. [Check Failed for ReadProtoFromBinaryFile (Failed to Parse NetParameter File)](#check-failed-for-readprotofrombinaryfile-failed-to-parse-netparameter-file)
        10. [3D OpenPose Returning Wrong Results: 0, NaN, Infinity, etc.](#3d-openpose-returning-wrong-results-0-nan-infinity-etc)
        11. [Protobuf Clip Param Caffe Error](#protobuf-clip-param-caffe-error)
        12. [The Human Skeleton Looks like Dotted Lines Rather than Solid Lines](#the-human-skeleton-looks-like-dotted-lines-rather-than-solid-lines)
        13. [CUDA_cublas_device_LIBRARY Not Found](#cuda_cublas_device_library-not-found)
        14. [CMake-GUI Error While Getting Default Caffe](#cmake-gui-error-while-getting-default-caffe)
        15. [Libgomp Out of Memory Error](#libgomp-out-of-memory-error)
        16. [Runtime Error with Turing GPU (Tesla T4) or Volta GPU](#runtime-error-with-turing-gpu-teslat4-or-volta-gpu)
        17. [Obscure CMake Error about Caffe or Pybind](#obscure-cmake-error-about-caffe-or-pybind).
    2. [Speed Performance Issues](#speed-performance-issues)
        1. [Speed Up, Memory Reduction, and Benchmark](#speed-up-memory-reduction-and-benchmark)
        2. [How to Measure the Latency Time?](#how-to-measure-the-latency-time)
        3. [CPU Version Too Slow](#cpu-version-too-slow)
        4. [Profiling Speed and Estimating FPS without Display](#profiling-speed-and-estimating-fps-without-display)
        5. [Webcam Slower than Images](#webcam-slower-than-images)
    3. [Accuracy Issues](#speed-performance-issues)
        1. [Is Maximum Accuracy Configuration Possible on Lower End GPUs?](#is-maximum-accuracy-configuration-possible-on-lower-end-gpus)
    4. [Other FAQ](#other-faq)
        1. [How Should I Link my IP Camera?](#how-should-i-link-my-ip-camera)
        2. [Difference between BODY_25 vs. COCO vs. MPI](#difference-between-body_25-vs-coco-vs-mpi)
        3. [Huge RAM Usage](#huge-ram-usage)





## FAQ
### Errors
#### Download Server Down
**Q: The download server is down, Download Hash missmatch, Error 502: Bad Gateway, etc. I.e., I cannot download the OpenPose models and/or 3rd party libraries.**
**A**: The are 2 alternatives. Option 1 (recommended): Download the links from [1602#issuecomment-641653411](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602#issuecomment-641653411) and download them in the places indicated by [doc/prerequisites.md](./prerequisites.md). Option 2: If you downloaded the models and dependencies to some local server, you could set the advanced CMake property `DOWNLOAD_SERVER` to link OpenPose to your local mirror server instead.



#### Out of Memory Error
**Q: Out of memory error** - I get an error similar to: `Check failed: error == cudaSuccess (2 vs. 0)  out of memory`.

**A**: Make sure you have a GPU with at least 4 GB of memory. If your GPU is between 2 and 4 GB, it should be fine for body-only settings, but you can also reduce the `--net_resolution` if it does not fit (check [Speed Up, Memory Reduction, and Benchmark](#speed-up-memory-reduction-and-benchmark) for the consequences of this).

(Only if you are compiling OpenPose by yourself, this does not apply to the portable OpenPose binaries for Windows because they already include cuDNN): If you have a GPU with >= 4 GB of memory, and you still face this error, most probably cuDNN is not installed/enabled. The default Caffe model uses >12 GB of GPU memory, cuDNN reduces it to ~2.2 GB for BODY_25 (default) and ~1.5 GB for COCO (`--model_pose COCO`). Note that you still need at least about 2.2 GB free for the default OpenPose to run. I.e., GPUs with only 2 GB will not fit the default OpenPose, and you will have to either switch to the `COCO` model (slower and less accurate), or reduce the `--net_resolution` (faster speed but also lower accuracy).

Also, hands and face increases the GPU memory requeriments, and 4 GB GPUs might run a bit short in some cases.



#### Video/Webcam Not Working
**Q: Video and/or webcam are not working** - Using a folder with images does work, but the video and/or the webcam do not. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers and video codecs (specially on Windows). Follow the same steps as the `Webcam is slow` question to test the webcam is working. After re-compiling OpenCV, you can also try this [OpenCV example for video](http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html).



#### Cannot Find OpenPose.dll Error (Windows)
**Q: System cannot find the file specified (Openpose.dll) error when trying to release** - Using a folder with images does work, but the video and/or the webcam do not. Note: often on Windows.

**A**: Visual Studio (VS) and the [doc/installation.md](./installation.md) section is only intended if you plan to modify the OpenPose code or integrate it with another library or project. If you just want to use the OpenPose demo, simply follow [doc/installation.md#windows-portable-demo](./installation.md#windows-portable-demo) and download the OpenPose binaries in the [Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) section.

If you need to compile it with Visual Studio (VS), then keep reading. In this error, VS is simply saying that there were errors while compiling the OpenPose library. Try compiling only the OpenPose library (not the demo), by right clicking on it, then `Set as StartUp Project`, and finally right click + `Build`. Then, at the bottom left part of VS, press `Error list` and then you should see which errors VS encountered while compiling. In that way, VS gives you the exact error so you can know it and share the exact issue.

If it didn't have any error, then setting OpenPoseDemo as main project again and F5 (or green play button) should work.

Note: OpenPose library is not an executable, but a library. So instead clicking F5 or the green button instead of `Build` will give you an error similar to `openpose.dll is not a valid Win32 application`.



#### Free Invalid Pointer Error
**Q: I am getting an error of the type: munmap_chunk()/free/invalid pointer.**

**A**: In order to run OpenCV 3.X and Caffe simultaneously, [OpenCV must be compiled without `WITH_GTK` and with `WITH_QT` flags](https://github.com/BVLC/caffe/issues/5282#issuecomment-306063718). On Ubuntu 16.04 the qt5 package is "qt5-default" and the OpenCV cmake option is WITH_QT.



#### Source Directory does not Contain CMakeLists.txt (Windows)
**Q: I am getting an error of the type: `The source directory {path to file} does not contain a CMakeLists.txt file.`.**

**A**: You might not have writing access to that folder. If you are in Windows, you should not try to install it in `Program Files`.



#### Always Zero People Detected
**Q: Always 0 people detected and displayed in default video and images.**

**A**: This answer assumes that never a single person is detected. If in your case it works sometimes, then check [Always Zero People Detected](#always-zero-people-detected). This always-0-people problem usually occurs in 2 situations: 1) When you selection `--num_gpu 0`, and 2) when the caffemodel has not been properly downloaded. E.g., if the connection drops when downloading the models.

For problem 1, setting `--num_gpu 0` means that no processing is done, so you can use this setting e.g., to record webcam. This functionality is kept for back-compatibility. You are most probably trying to run on CPU-only mode, for that, install OpenPose in CPU-only mode following [doc/installation.md](./installation.md).

For problem 2, try the following solutions (in this order):

1. Assuming that default OpenPose (i.e., BODY_25 model) failed, try with `--model_pose COCO` and `--model_pose MPII` models. If any of them work, the `caffemodel` files of the other models were corrupted while being downloaded. Otherwise, it will most probably be a Caffe/protobuf issue.
2. Assuming that the model is corrupted, remove the current models in the model folder, and download them manually from the links in [doc/installation.md](./installation.md). Alternatively, remove them and re-run Cmake again. If this does not work, try downloading the COCO_25 model from the browser following the download link on this [Dropbox link](https://www.dropbox.com/s/03r8pa8sikrqv62/pose_iter_584000.caffemodel).
3. If none of the OpenPose models are working, make sure Caffe is working properly and that you can run the Caffe examples with other caffemodel / prototxt files.



#### Very Few People Detected
**Q: Low detection rate. It can detect the person on some images (usually higher contrast, with bigger people), but it will fail for most of images with low resolution or small people.**

**A**: Images with low resolution, or with people too tiny will simply not work too well. However, it can be highly improved by using the maximum accuracy configuration detailed in [doc/quick_start.md#maximum-accuracy-configuration](./quick_start.md#maximum-accuracy-configuration).



#### Check Failed for ReadProtoFromBinaryFile (Failed to Parse NetParameter File)
**Q: I am facing an error similar to:** `Check failed: ReadProtoFromBinaryFile(param_file, param) Failed to parse NetParameter file: models/pose/body_25/pose_iter_584000.caffemodel`

**A**: Same answer than for [Always Zero People Detected](#always-zero-people-detected).



#### 3D OpenPose Returning Wrong Results: 0, NaN, Infinity, etc.
**Q: 3D OpenPose is returning wrong results.**

**A**: 99.99% of the cases, this is due to wrong or poor calibration. Repeat the calibration making sure that the final reprojection error is about or less than 0.1 pixels.



#### Protobuf Clip Param Caffe Error
**Q: Runtime error similar to:**
```
[libprotobuf ERROR google/protobuf/message_lite.cc:123] Can't parse message of type "caffe.NetParameter" because it is missing required fields: layer[0].clip_param.min, layer[0].clip_param.max
F0821 14:26:29.665053 22812 upgrade_proto.cpp:97] Check failed: ReadProtoFromBinaryFile(param_file, param) Failed to parse NetParameter file: models/pose/body_25/pose_iter_584000.caffemodel
```

**A**: This error has been solved in the latest OpenPose versions. Completely remove OpenPose and re-download the latest version (just cleaning the compilation or removing the `build/` folder will not work).

If you wanna use your custom Caffe and it has this error: This error only happens in some Ubuntu machines. Following #787, compile your own Caffe with an older version of it. The hacky (quick but not recommended way) is to follow [#787#issuecomment-415476837](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/787#issuecomment-415476837), the elegant way (compatible with future OpenPose versions) is to build your own Caffe independently, following [doc/installation.md#custom-caffe-ubuntu-only](./installation.md#custom-caffe-ubuntu-only).



#### The Human Skeleton Looks like Dotted Lines Rather than Solid Lines
**Q**: When I use the demo to handle my images，the skeletons are dotted lines. I want to know how to make them to be solid lines.

**A**: The reason is that your input image size is too small. You can either 1) manually rescale your images up or 2) use a bigger `--output_resolution` so OpenPose will resize them up.



#### CUDA_cublas_device_LIBRARY Not Found
**Q**: I encounter an error similar to the following:
```
CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
Please set them or make sure they are set and tested correctly in the CMake files:
CUDA_cublas_device_LIBRARY (ADVANCED)
    linked by target "caffe" in directory /home/jakebmalis/Documents/openpose/3rdparty/caffe/src/caffe
```

**A**: Make sure to download and install CMake-GUI following the [doc/prerequisites.md](./prerequisites.md) section. This is a known problem with CMake-GUI versions from 3.8 to 3.11 (unfortunately, default Ubuntu 18 CMake-GUI uses 3.10). You will need a CMake version >= 3.12.



#### CMake-GUI Error While Getting Default Caffe
**Q**: It seems to me CMake-gui does not download Caffe at all. I tried to wipe everything and try to install OpenPose again, but received the same mistake. I also tried to see if cmake follows the ifs in the CMakeLists.txt correctly and reaches the branches where he establishes that Caffe needs to be downloaded and it seems to me it does so.

**A**: There are 2 solutions to try. First, if you were using an old OP version and you just updated it, you should simply completely remove that OpenPose folder, and then re-download and re-compile OpenPose. Second, and only if after re-cloning master and running CMake-GUI the `3rdparty/caffe/` folder stays empty, manually trigger the git submodules to update. So the clone step becomes:
```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose

git submodule init
git submodule update
```




#### Libgomp Out of Memory Error
**Q**: When I start OpenPose, I receive an error similar to the following: `libgomp: Out of memory allocating 927712937064 bytes`.

**A**: Reinstall and upgrade from scratch CUDA, cuDNN, Python, and OpenCV (GitHub issue #1160).




#### Runtime Error with Turing GPU (Tesla T4) or Volta GPU
**Q**: When I start OpenPose, I receive a runtime error for new GPU architectures.

**A**: To solve this problem, 1) make sure you are using CUDA 10 or higher, and 2) change line 7 in `{OPENPOSE_PATH}/3rdparty/caffe/cmake/Cuda.cmake`, from `set(Caffe_known_gpu_archs "30 35 50 52 60 61")` to `set(Caffe_known_gpu_archs "30 35 50 52 60 61 75")`.




#### Obscure CMake Error about Caffe or Pybind
**Q:** There appear some weird and obscure errors on CMake about Caffe and/or Pybind.

**A**: Check [doc/installation.md](./installation.md) to run the `git submodule` command, i.e.,
```
git submodule update --init --recursive --remote
```





### Speed Performance Issues
#### Speed Up, Memory Reduction, and Benchmark
**Q: Low speed** - OpenPose is quite slow, is it normal? How can I speed it up?

**A**: Check [doc/speed_up_openpose.md](./speed_up_openpose.md) to discover the approximate speed of your graphics card and some speed tips.



#### How to Measure the Latency Time?
**Q: How to measure/calculate/estimate the latency/lag time?**

**A**: [Profile](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/speed_up_openpose.md#profiling-speed) the OpenPose speed. For 1-GPU or CPU-only systems (use `--disable_multi_thread` for simplicity in multi-GPU systems for latency measurement), the latency will be roughly the sum of all the reported measurements.



#### CPU Version Too Slow
**Q: The CPU version is insanely slow compared to the GPU version.**

**A**: Check [doc/speed_up_openpose.md#cpu-version](./speed_up_openpose.md#cpu-version) to discover the approximate speed and some speed tips.



#### Profiling Speed and Estimating FPS without Display
Check the [doc/speed_up_openpose.md#profiling-speed](./speed_up_openpose.md#profiling-speed) section.



#### Webcam Slower than Images
**Q: Webcam is slow** - Using a folder with images matches the speed FPS benchmarks, but the webcam has lower FPS. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers (specially on Windows). The first step should be to compile OpenCV by your own and re-compile OpenPose after that (following the [doc/installation.md#reinstallation](./installation.md#reinstallation) section). If the speed is still slower, you can better debug it by running a webcam OpenCV example (e.g. [this C++ example](http://answers.opencv.org/question/1/how-can-i-get-frames-from-my-webcam/)). If you are able to get the proper FPS with the OpenCV demo but OpenPose is still low, then let us know!





### Accuracy Issues
#### Is Maximum Accuracy Configuration Possible on Lower End GPUs?
**Q**: I've read that this command provides the most accurate results possible on Openpose so far: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md#maximum-accuracy-configuration. However, a 8GB GPU (e.g., 1080 or 2080) will run out of memory, is there any method to achieve the same accuracy on GPU using less memory even if it meant sacrificing speed?

**A**: Unfortunately no, there is no way at the moment. Caffe just takes so much memory doing that. You can try with `--scale_number 3` instead of 4, reducing a bit the `net_resolution` (e.g. `720` vs. `736`) and starting the computer without GUI (which also takes about 1GB of memory just to keep the computer GUI running).





### Other FAQ
#### How Should I Link my IP Camera?
**Q: How Should I Link my IP Camera with http protocol?**

**A**: Usually with `http://CamIP:PORT_NO./video?x.mjpeg`.



#### Difference between BODY_25 vs. COCO vs. MPI
COCO model will eventually be removed. BODY_25 model is faster, more accurate, and it includes foot keypoints. However, COCO requires less memory on GPU (being able to fit into 2GB GPUs with the default settings) and it runs faster on CPU-only mode. MPI model is only meant for people requiring the MPI-keypoint structure. It is also slower than BODY_25 and far less accurate.



#### Huge RAM Usage
**Q**: During the execution of the demo, the CPU usage oscillates between 50 and 99%, but the RAM is almost at 100%.

**A**: Highly reducing the `--net_resolution` and following the tips in the [Speed Up, Memory Reduction, and Benchmark](#speed-up-memory-reduction-and-benchmark) section are the only way to reduce the RAM usage. Alternatively, you can disable `USE_MKL` in CMake, which will highly reduce the RAM usage of the Caffe version, but it might also reduce the program speed.
