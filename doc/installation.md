OpenPose Library - Compilation and Installation
====================================

## Requirements
- Ubuntu (tested on 14 and 16)
- GPU with **cuDNN installed** and at least 2 GB and 1.5 GB available (the `nvidia-smi` command checks the available GPU memory in Ubuntu).
- At least 2 GB of free RAM memory.
- Highly recommended: A CPU with at least 8 cores. 

Note: This requirements assume the default configuration (i.e. `--net_resolution "656x368"` and `num_scales 1`). You might need more (with a greater net resolution and/or number of scales) or less resources (with smaller net resolution and/or using the MPI and MPI_4 models).

## How to Compile It
1. Required: CUDA, cuDNN and OpenCV installed on your machine. OpenCV can be easily installed by running `apt-get install libopencv-dev`.
2. If you have installed OpenCV 2.4 in your system, go to step 3. If you are using OpenCV 3, uncomment the line `# OPENCV_VERSION := 3` on the file `Makefile.config.Ubuntu14.example` (for Ubuntu 14) and/or `Makefile.config.Ubuntu16.example` (for Ubuntu 15 or 16). In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have manually installed it and you need to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
3. Build Caffe & the OpenPose library + download the required Caffe models for Ubuntu 14.04 or 16.04 and cuda 8:
```
chmod u+x install_caffe_and_openpose.sh
./install_caffe_and_openpose.sh
```

Alternatively, if you want to use CUDA 7 and/or avoid using sh scripts, install the Caffe prerequisites and then run these lines:
```
### Install Caffe ###
cd 3rdparty/caffe/
# Select your desired Makefile file (run only one of the next 4 commands)
cp Makefile.config.Ubuntu14_cuda_7.example Makefile.config # Ubuntu 14, cuda 7
cp Makefile.config.Ubuntu14.example Makefile.config # Ubuntu 14, cuda 8
cp Makefile.config.Ubuntu16_cuda_7.example Makefile.config # Ubuntu 16, cuda 7
cp Makefile.config.Ubuntu16.example Makefile.config # Ubuntu 16, cuda 8
# Compile Caffe
make distribute -j${number_of_cpus}

### Install OpenPose ###
cd ../../models/
./getModels.sh # It just downloads the Caffe trained models
cd ..
# Same file cp command as the one used for Caffe
cp Makefile.config.Ubuntu14_cuda_7.example Makefile.config
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

## Custom Caffe
We slightly modified some Caffe compilation flags and minor details. In case you want to use your own Caffe distribution, these are the files we added and modified:

1. Added files: `install_caffe_and_openpose.sh`; as well as `Makefile.config.Ubuntu14.example`, `Makefile.config.Ubuntu16.example`, `Makefile.config.Ubuntu14_cuda_7.example` and `Makefile.config.Ubuntu16_cuda_7.example` (extracted from `Makefile.config.example`).
2. Optional - deleted Caffe files and folders (only to save space): `Makefile.config.example`, `data/`, `examples/` (editing the Makefile distribute section to avoid using this folder) and `models/`.
3. Finally, run `make distribute` in your Caffe version and modify the Caffe directory variable in our Makefile config file: `./Makefile.config.UbuntuX.example` (where X is 14 or 16 depending on your Ubuntu version), set the `CAFFE_DIR` parameter to the path where both the `include` and `lib` Caffe folders are located.
