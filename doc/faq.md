OpenPose - Frequently Asked Question (FAQ)
============================================

## Contents
1. [FAQ](#faq)
    1. [Out of Memory Error](#out-of-memory-error)
    2. [Speed Up and Benchmark](#speed-up-and-benchmark)
    3. [Estimating FPS without Display](#estimating-fps-without-display)
    4. [Webcam Slower than Images](#webcam-slower-than-images)
    5. [Video/Webcam Not Working](#videowebcam-not-working)
    6. [Cannot Find OpenPose.dll Error](#cannot-find-openpose.dll-error-windows)
    7. [Free Invalid Pointer Error](#free-invalid-pointer-error)
    8. [Source Directory does not Contain CMakeLists.txt (Windows)](#source-directory-does-not-contain-cmakelists.txt-windows)
    9. [How Should I Link my IP Camera?](#how-should-i-link-my-ip-camera)





## FAQ
### Out of Memory Error
**Q: Out of memory error** - I get an error similar to: `Check failed: error == cudaSuccess (2 vs. 0)  out of memory`.

**A**: Most probably cuDNN is not installed/enabled, the default Caffe model uses >12 GB of GPU memory, cuDNN reduces it to ~1.5 GB.



### Speed Up and Benchmark
**Q: Low speed** - OpenPose is quite slow, is it normal? How can I speed it up?

**A**: Check the [OpenPose Benchmark](https://docs.google.com/spreadsheets/d/1-DynFGvoScvfWDA1P4jDInCkbD4lg0IKOYbXgEq0sK0/edit#gid=0) to discover the approximate speed of your graphics card. Some speed tips:

    1. Use cuDNN 5.1 (cuDNN 6 is ~10% slower).
    2. Reduce the `--net_resolution` (e.g. to 320x176) (lower accuracy).
    3. For face, reduce the `--face_net_resolution`. The resolution 320x320 usually works pretty decently.
    4. Use the `MPI_4_layers` model (lower accuracy and lower number of parts).
    5. Change GPU rendering by CPU rendering to get approximately +0.5 FPS (`--render_pose 1`).



### Estimating FPS without Display
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
**Q: How Should I Link my IP Camera with http protocol?.**

**A**: Usually with `http://CamIP:PORT_NO./video?x.mjpeg`.
