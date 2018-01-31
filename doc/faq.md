OpenPose - Frequently Asked Question (FAQ)
============================================

## Contents
1. [FAQ](#faq)
    1. [Out of Memory Error](#out-of-memory-error)
    2. [Speed Up and Benchmark](#speed-up-and-benchmark)
    3. [Webcam Slower than Images](#webcam-slower-than-images)
    4. [Vide/Webcam Not Working](#video-webcam-not-working)
    5. [Free Invalid Pointer Error](#free-invalid-pointer-error)





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



### Webcam Slower than Images
**Q: Webcam is slow** - Using a folder with images matches the speed FPS benchmarks, but the webcam has lower FPS. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers (specially on Windows). The first step should be to compile OpenCV by your own and re-compile OpenPose after that (following the `Reinstallation` section in Ubuntu or cleaning the project on Windows). If the speed is still slower, you can better debug it by running a webcam OpenCV example (e.g. [this C++ example](http://answers.opencv.org/question/1/how-can-i-get-frames-from-my-webcam/)). If you are able to get the proper FPS with the OpenCV demo but OpenPose is still low, then let us know!



### Video/Webcam Not Working
**Q: Video and/or webcam are not working** - Using a folder with images does work, but the video and/or the webcam do not. Note: often on Windows.

**A**: OpenCV has some issues with some camera drivers and video codecs (specially on Windows). Follow the same steps as the `Webcam is slow` question to test the webcam is working. After re-compiling OpenCV, you can also try this [OpenCV example for video](http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html).



### Free Invalid Pointer Error
**Q: I am getting an error of the type: munmap_chunk()/free/invalid pointer.**

**A**: In order to run OpenCV 3.X and Caffe simultaneously, [OpenCV must be compiled without `WITH_GTK` and with `WITH_QT` flags](https://github.com/BVLC/caffe/issues/5282#issuecomment-306063718). On Ubuntu 16.04 the qt5 package is "qt5-default" and the OpenCV cmake option is WITH_QT.
