OpenPose Library - Compilation and Installation
====================================



## Requirements
- Ubuntu (tested on 14 and 16)
- GPU with at least 2 GB and 1.5 GB available (the `nvidia-smi` command checks the available GPU memory in Ubuntu).
- CUDA and cuDNN installed.
- At least 2 GB of free RAM memory.
- Highly recommended: A CPU with at least 8 cores.

Note: These requirements assume the default configuration (i.e. `--net_resolution "656x368"` and `num_scales 1`). You might need more (with a greater net resolution and/or number of scales) or less resources (with smaller net resolution and/or using the MPI and MPI_4 models).



## Installation
### Script Compilation
**Highly important**: This script only works with CUDA 8 and Ubuntu 14 or 16. Otherwise, check [Manual Compilation](#manual-compilation).
1. Required: CUDA, cuDNN, OpenCV and Atlas must be already installed on your machine.
    1. OpenCV can be installed with `apt-get install libopencv-dev`. If you have compiled OpenCV 3 by your own, follow [Manual Compilation](#manual-compilation). After both Makefile.config files have been generated, edit them and uncomment the line `# OPENCV_VERSION := 3`. You might alternatively modify all `Makefile.config.UbuntuXX` files and then run the scripts in step 2.
    2. In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have OpenCV 3 compiled with the contrib module and you want to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
    3. Atlas can be installed with `sudo apt-get install libatlas-base-dev`. Instead of Atlas, you can use OpenBLAS or Intel MKL by modifying the line `BLAS := atlas` in the same way as previosuly mentioned for the OpenCV version selection.
2. Build Caffe & the OpenPose library + download the required Caffe models for Ubuntu 14.04 or 16.04 (auto-detected for the script) and CUDA 8:
```
chmod u+x install_caffe_and_openpose.sh
./install_caffe_and_openpose.sh
```

### Manual Compilation
Alternatively to the script installation, if you want to use CUDA 7, avoid using sh scripts, change some configuration labels (e.g. OpenCV version), etc., then:
1. Install the [Caffe prerequisites](http://caffe.berkeleyvision.org/installation.html).
2. Compile Caffe and OpenPose by running these lines:
    ```
    ### Install Caffe ###
    cd 3rdparty/caffe/
    # Select your desired Makefile file (run only one of the next 4 commands)
    cp Makefile.config.Ubuntu14_cuda_7.example Makefile.config # Ubuntu 14, cuda 7
    cp Makefile.config.Ubuntu14.example Makefile.config # Ubuntu 14, cuda 8
    cp Makefile.config.Ubuntu16_cuda_7.example Makefile.config # Ubuntu 16, cuda 7
    cp Makefile.config.Ubuntu16.example Makefile.config # Ubuntu 16, cuda 8
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    # Compile Caffe
    make all -j${number_of_cpus} && make distribute -j${number_of_cpus}

    ### Install OpenPose ###
    cd ../../models/
    ./getModels.sh # It just downloads the Caffe trained models
    cd ..
    # Same file cp command as the one used for Caffe
    cp Makefile.config.Ubuntu14_cuda_7.example Makefile.config
    # Change any custom flag from the resulting Makefile.config (e.g. OpenCV 3, Atlas/OpenBLAS/MKL, etc.)
    make all -j${number_of_cpus}
    ```

    NOTE: If you want to use your own Caffe distribution, follow the steps on `Custom Caffe` section and later re-compile the OpenPose library:
    ```
    chmod u+x install_openpose.sh
    ./install_openpose.sh
    ```
    Note: These steps only need to be performed once. If you are interested in making changes to the OpenPose library, you can simply recompile it with:
    ```
    make clean
    make all -j$(NUM_CORES)
    ```
**Highly important**: There are 2 `Makefile.config.Ubuntu##.example` analogous files, one in the main folder and one in [3rdparty/caffe/](../3rdparty/caffe/), corresponding to OpenPose and Caffe configuration files respectively. Any change must be done to both files (e.g. OpenCV 3 flag, Atlab/OpenBLAS/MKL flag, etc.). E.g. for CUDA 8 and Ubuntu16: [3rdparty/caffe/Makefile.config.Ubuntu16.example](../3rdparty/caffe/Makefile.config.Ubuntu16.example) and [Makefile.config.Ubuntu16.example](../Makefile.config.Ubuntu16.example).




## Reinstallation
If you updated some software that our library or 3rdparty use, or you simply want to reinstall it:
1. Clean the OpenPose and Caffe compilation folders:
```
make clean && cd 3rdparty/caffe && make clean
```
2. Repeat the [Installation](#installation) steps.



## Uninstallation
You just need to remove the OpenPose folder, by default called `openpose/`. E.g. `rm -rf openpose/`.
