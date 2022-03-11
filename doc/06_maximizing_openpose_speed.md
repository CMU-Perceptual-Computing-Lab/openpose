OpenPose Doc - Maximizing the OpenPose Speed
========================================================================================

## Contents
1. [OpenPose Benchmark](#openpose-benchmark)
2. [Profiling Speed](#profiling-speed)
3. [CPU Version](#cpu-version)
4. [Speed Up Preserving Accuracy](#speed-up-preserving-accuracy)
5. [Speed Up and Memory Reduction](#speed-up-and-memory-reduction)





## OpenPose Benchmark
Check the [OpenPose Benchmark](https://docs.google.com/spreadsheets/d/1-DynFGvoScvfWDA1P4jDInCkbD4lg0IKOYbXgEq0sK0/edit#gid=0) to discover the approximate expected speed of your graphics card.



### CPU Version
The CPU version runs at about 0.3 FPS on the COCO model, and at about 0.1 FPS (i.e., about 15 sec / frame) on the default BODY_25 model. Switch to COCO model and/or reduce the `net_resolution` as indicated above. Contradictory fact: BODY_25 model is about 5x slower than COCO on CPU-only version, but it is about 40% faster on GPU version.

On Ubuntu (for OS versions older than 20), you can also boost CPU-only speed by 2-3x by following [installation/0_index.md#faster-cpu-version-ubuntu-only](installation/0_index.md#faster-cpu-version-ubuntu-only).



### Profiling Speed
OpenPose displays the FPS in the basic GUI. However, more complex speed metrics can be obtained from the command line while running OpenPose. In order to obtain those, compile OpenPose with the `PROFILER_ENABLED` flag on CMake-gui. OpenPose will automatically display time measurements for each subthread after processing `F` frames (by default `F = 1000`, but it can be modified with the `--profile_speed` flag, e.g. `--profile_speed 100`).

- Time measurement for 1 graphic card: The FPS will be the slowest time displayed in your terminal command line (as OpenPose is multi-threaded). Times are in milliseconds, so `FPS = 1000/millisecond_measurement`.
- Time measurement for >1 graphic cards: Assuming `n` graphic cards, you will have to wait up to `n` x `F` frames to visualize each graphic card speed (as the frames are split among them). In addition, the FPS would be: `FPS = minFPS(speed_per_GPU/n, worst_time_measurement_other_than_GPUs)`. For < 4 GPUs, this is usually `FPS = speed_per_GPU/n`.

Make sure that `wPoseExtractor` time is the slowest timing. Otherwise the input producer (video/webcam codecs issues with OpenCV, images too big, etc.) or the GUI display (use OpenGL support as detailed in the next section (`Speed Up Preserving Accuracy`) might not be optimized.



## Speed Up Preserving Accuracy
Some speed tips to maximize the OpenPose runtime speed while preserving the accuracy (do not expect miracles, but it might help a bit boosting the framerate):

    1. Enable the `WITH_OPENCV_WITH_OPENGL` flag in CMake to have a much faster GUI display. It reduces the lag and increase the speed of displaying images by telling OpenCV to render the images using OpenGL support. This speeds up display rendering about 3x. E.g., it reduces from about 30 msec to about 3-10 msec the display time for HD resolution images. It requires OpenCV to be compiled with OpenGL support and it provokes a visual aspect-ratio artifact when rendering a folder with images of different resolutions. Note: Default OpenCV in Ubuntu 16 (from apt-get install) does have OpenGL support included. Nevertheless, default one from Ubuntu 18 and the Windows portable binaries do not.
    2. Change GPU rendering by CPU rendering to get approximately +0.5 FPS (`--render_pose 1`).
    3. Use cuDNN 5.1 or 7.2 (cuDNN 6 is ~10% slower).
    4. Use the `BODY_25` model for simultaneously maximum speed and accuracy (both COCO and MPII models are slower and less accurate). But it does increase the GPU memory, so it might go out of memory more easily in low-memory GPUs.
    5. Enable the AVX flag in CMake-GUI (if your computer supports it).



## Speed Up and Memory Reduction
Some speed tips to highly maximize the OpenPose speed, but keep in mind the accuracy trade-off:

    1. Reduce the `--net_resolution` (e.g., to 320x176) (lower accuracy). Note: For maximum accuracy, follow [doc/01_demo.md#maximum-accuracy-configuration](01_demo.md#maximum-accuracy-configuration).
    2. For face, reduce the `--face_net_resolution`. The resolution 320x320 usually works pretty decently.
    3. Points 1-2 will also reduce the GPU memory usage (or RAM memory for CPU version).
    4. Use the `BODY_25` model for maximum speed. Use `MPI_4_layers` model for minimum GPU memory usage (but lower accuracy, speed, and number of parts).
