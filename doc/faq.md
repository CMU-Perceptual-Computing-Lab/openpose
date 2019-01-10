OpenPose - Frequently Asked Question (FAQ)
============================================

## Contents
1. [FAQ](#faq)
    1. [Out of Memory Error](#out-of-memory-error)
    2. [Speed Up, Memory Reduction, and Benchmark](#speed-up-memory-reduction-and-benchmark)
    3. [CPU Version Too Slow](#cpu-version-too-slow)
    4. [Profiling Speed and Estimating FPS without Display](#profiling-speed-and-estimating-fps-without-display)
    5. [Webcam Slower than Images](#webcam-slower-than-images)
    6. [Video/Webcam Not Working](#videowebcam-not-working)
    7. [Cannot Find OpenPose.dll Error](#cannot-find-openpose.dll-error-windows)
    8. [Free Invalid Pointer Error](#free-invalid-pointer-error)
    9. [Source Directory does not Contain CMakeLists.txt (Windows)](#source-directory-does-not-contain-cmakelists.txt-windows)
    10. [How Should I Link my IP Camera?](#how-should-i-link-my-ip-camera)
    11. [Difference between BODY_25 vs. COCO vs. MPI](#difference-between-body_25-vs.-coco-vs.-mpi)
    12. [How to Measure the Latency Time?](#how-to-measure-the-latency-time)
    13. [Zero People Detected](#zero-people-detected)
    14. [Check Failed for ReadProtoFromBinaryFile (Failed to Parse NetParameter File)](#check-failed-for-readprotofrombinaryfile-failed-to-parse-netparameter-file)
    15. [3D OpenPose Returning Wrong Results: 0, NaN, Infinity, etc.](#3d-openpose-returning-wrong-results-0-nan-infinity-etc)
    16. [Protobuf Clip Param Caffe Error](#protobuf-clip-param-caffe-error)





## FAQ
### Out of Memory Error
**Q: Out of memory error** - I get an error similar to: `Check failed: error == cudaSuccess (2 vs. 0)  out of memory`.

**A**: Most probably cuDNN is not installed/enabled, the default Caffe model uses >12 GB of GPU memory, cuDNN reduces it to ~2 GB for BODY_25 and ~1.5 GB for COCO.



### Speed Up, Memory Reduction, and Benchmark
**Q: Low speed** - OpenPose is quite slow, is it normal? How can I speed it up?

**A**: Check [doc/speed_up_preserving_accuracy.md](./speed_up_preserving_accuracy.md) to discover the approximate speed of your graphics card and some speed tips.



### CPU Version Too Slow
**Q: The CPU version is insanely slow compared to the GPU version.**

**A**: Check [doc/speed_up_preserving_accuracy.md#cpu-version](./speed_up_preserving_accuracy.md#cpu-version) to discover the approximate speed and some speed tips.



### Profiling Speed and Estimating FPS without Display
Check the [doc/installation.md#profiling-speed](./installation.md#profiling-speed) section.



### Webcam Slower than Images
**Q: Webcam is slow** - Using a folder with images matches the speed FPS benchmarks, but the webcam has lower FPS. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers (specially on Windows). The first step should be to compile OpenCV by your own and re-compile OpenPose after that (following the [doc/installation.md#reinstallation](./installation.md#reinstallation) section). If the speed is still slower, you can better debug it by running a webcam OpenCV example (e.g. [this C++ example](http://answers.opencv.org/question/1/how-can-i-get-frames-from-my-webcam/)). If you are able to get the proper FPS with the OpenCV demo but OpenPose is still low, then let us know!



### Video/Webcam Not Working
**Q: Video and/or webcam are not working** - Using a folder with images does work, but the video and/or the webcam do not. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers and video codecs (specially on Windows). Follow the same steps as the `Webcam is slow` question to test the webcam is working. After re-compiling OpenCV, you can also try this [OpenCV example for video](http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html).



### Cannot Find OpenPose.dll Error (Windows)
**Q: System cannot find the file specified (Openpose.dll) error when trying to release** - Using a folder with images does work, but the video and/or the webcam do not. Note: often on Windows.

**A**: Visual Studio (VS) and the [doc/installation.md](./installation.md) section is only intended if you plan to modify the OpenPose code or integrate it with another library or project. If you just want to use the OpenPose demo, simply follow [doc/installation.md#windows-portable-demo](./installation.md#windows-portable-demo) and download the OpenPose binaries in the [Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) section.

If you need to compile it with Visual Studio (VS), then keep reading. In this error, VS is simply saying that there were errors while compiling the OpenPose library. Try compiling only the OpenPose library (not the demo), by right clicking on it, then `Set as StartUp Project`, and finally right click + `Build`. Then, at the bottom left part of VS, press `Error list` and then you should see which errors VS encountered while compiling. In that way, VS gives you the exact error so you can know it and share the exact issue.

If it didn't have any error, then setting OpenPoseDemo as main project again and F5 (or green play button) should work.

Note: OpenPose library is not an executable, but a library. So instead clicking F5 or the green button instead of `Build` will give you an error similar to `openpose.dll is not a valid Win32 application`.



### Free Invalid Pointer Error
**Q: I am getting an error of the type: munmap_chunk()/free/invalid pointer.**

**A**: In order to run OpenCV 3.X and Caffe simultaneously, [OpenCV must be compiled without `WITH_GTK` and with `WITH_QT` flags](https://github.com/BVLC/caffe/issues/5282#issuecomment-306063718). On Ubuntu 16.04 the qt5 package is "qt5-default" and the OpenCV cmake option is WITH_QT.



### Source Directory does not Contain CMakeLists.txt (Windows)
**Q: I am getting an error of the type: `The source directory {path to file} does not contain a CMakeLists.txt file.`.**

**A**: You might not have writing access to that folder. If you are in Windows, you should not try to install it in `Program Files`.



### How Should I Link my IP Camera?
**Q: How Should I Link my IP Camera with http protocol?**

**A**: Usually with `http://CamIP:PORT_NO./video?x.mjpeg`.



### Difference between BODY_25 vs. COCO vs. MPI
COCO model will eventually be removed. BODY_25 model is faster, more accurate, and it includes foot keypoints. However, COCO requires less memory on GPU (being able to fit into 2GB GPUs with the default settings) and it runs faster on CPU-only mode. MPI model is only meant for people requiring the MPI-keypoint structure. It is also slower than BODY_25 and far less accurate.



### How to Measure the Latency Time?
**Q: How to measure/calculate/estimate the latency/lag time?**

**A**: [Profile](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#profiling-speed) the OpenPose speed. For 1-GPU or CPU-only systems (use `--disable_multi_thread` for simplicity in multi-GPU systems for latency measurement), the latency will be roughly the sum of all the reported measurements.



### Zero People Detected
**Q: 0 people detected and displayed in default video and images.**

**A**: This problem usually occurs when the caffemodel has not been properly downloaded. E.g., if the connection drops when downloading the models. Try the following solutions (in this order):

1. Assuming that default OpenPose (i.e., BODY_25 model) failed, try with `--model_pose COCO` and `--model_pose MPII` models. If any of them work, the `caffemodel` files of the other models were corrupted while being downloaded. Otherwise, it will most probably be a Caffe/protobuf issue.
2. Assuming that the model is corrupted, remove the current models in the model folder, and download them manually from the links in [doc/installation.md](./installation.md). Alternatively, remove them and re-run Cmake again. If this does not work, try downloading the COCO_25 model from the browser following the download link on this [Dropbox link](https://www.dropbox.com/s/03r8pa8sikrqv62/pose_iter_584000.caffemodel).
3. If none of the OpenPose models are working, make sure Caffe is working properly and that you can run the Caffe examples with other caffemodel / prototxt files.



### Check Failed for ReadProtoFromBinaryFile (Failed to Parse NetParameter File)
**Q: I am facing an error similar to:** `Check failed: ReadProtoFromBinaryFile(param_file, param) Failed to parse NetParameter file: models/pose/body_25/pose_iter_584000.caffemodel`

**A**: Same answer than for [Zero People Detected](#zero-people-detected).



### 3D OpenPose Returning Wrong Results: 0, NaN, Infinity, etc.
**Q: 3D OpenPose is returning wrong results.**

**A**: 99.99% of the cases, this is due to wrong or poor calibration. Repeat the calibration making sure that the final reprojection error is about or less than 0.1 pixels.



### Protobuf Clip Param Caffe Error
**Q: Runtime error similar to:**
```
[libprotobuf ERROR google/protobuf/message_lite.cc:123] Can't parse message of type "caffe.NetParameter" because it is missing required fields: layer[0].clip_param.min, layer[0].clip_param.max
F0821 14:26:29.665053 22812 upgrade_proto.cpp:97] Check failed: ReadProtoFromBinaryFile(param_file, param) Failed to parse NetParameter file: models/pose/body_25/pose_iter_584000.caffemodel
```

**A**: This error only happens in some Ubuntu machines. Following #787, compile your own Caffe with an older version of it. The hacky (quick but not recommended way) is to follow [#787#issuecomment-415476837](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/787#issuecomment-415476837), the elegant way (compatible with future OpenPose versions) is to build your own Caffe independently, following [doc/installation.md#custom-caffe-ubuntu-only](./installation.md#custom-caffe-ubuntu-only).

Note that OpenPose uses a [custom fork of Caffe](https://github.com/CMU-Perceptual-Computing-Lab/caffe) (rather than the official Caffe master), which it is only updated if it works on our machines. Currently, this version works on a newly formatted machine (Ubuntu 16.04 LTS) and in all our machines (CUDA 8 and 10 tested). The default GPU version is the master branch, which it is also compatible with CUDA 10 without changes (official Caffe version requires some changes for it). We also use the OpenCL and CPU tags if their CMake flags are selected.
