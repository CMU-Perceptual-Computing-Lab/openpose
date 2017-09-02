# Installation using CMake

The instructions in this section describe the steps to build OpenPose using CMake. Currently, CMake support has only
been tested on Ubuntu OS.

## Clone the repository

The first step is to clone the OpenPose repository from GitHub.

```bash
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```

## Install the prerequisites

Since currently, OpenPose uses Caffe under the hood -- most prerequisites are for Caffe itself. If Caffe is already installed, the user may skip installing most of these packages. The packages can be installed using the script `install_cmake.sh` in the `ubuntu` directory.

## Generate the makefiles

There are two ways to generate the makefiles, either using CMake GUI program or the command line.Both are described
succinctly in the sections below.

### CMake GUI

* The first step is to open the CMake GUI.
 ![im_1](media/cmake_installation/im_1.png)

* After opening the CMake GUI, the next step is to select the project source directory and a sub-directory where the makefiles will
be generated. We will first select the openpose directory and then we will select a `build` directory in the project root directory as shown in the image below (See the red rectangle). If the `build` directory does not exists, CMake will create one for us.

![im_2](media/cmake_installation/im_2.png)

* Next press the `Configure` button in the GUI. It will first ask you to create the `build` directory, if it already did not exist. Press `Yes`.

![im_3](media/cmake_installation/im_3.png)

* Next a new dialog box will appear, press the `Finish` button here.

![im_4](media/cmake_installation/im_4.png)

* If it shows an error as shown below. It is perfect alright -- we'll fix it in the next step. If not, skip to building
OpenPose.

![im_5](media/cmake_installation/im_5.png)

* The error probably occurred because CMake could not find the Caffe includes and the library. 

#### Caffe already present 

* If Caffe is already installed, specify the Caffe includes path and the library as shown below. 

![im_6](media/cmake_installation/im_6.png)

* To generate the makefile, press the `Generate` button and proceed to building OpenPose.

![im_7](media/cmake_installation/im_7.png)

#### Caffe not present

* If Caffe is not already there, the OpenPose build will do it for you. Just turn on the `BUILD_CAFFE` key as shown below.

![im_8](media/cmake_installation/im_8.png)

* To generate the makefile, press the `Generate` button and proceed to building OpenPose.

![im_9](media/cmake_installation/im_9.png)

SIDENOTE -- If you have OpenCV installed from source -- you can specify it using the `OPENCV_DIR` variable to the
directory where you build OpenCV.

### Command Line build

After cloning the next step is to create a build folder where we will build the library --

```bash
cd openpose
mkdir build
cd build
```

The next step is to generate the makefiles. Now there can be multiple scenarios based on what the user already has e.x.
Caffe might be already installed and the user might be interested in building OpenPose against the that version of Caffe
instead of requiring OpenPose to build Caffe from scratch.

#### SCENARIO 1 -- Caffe not installed and Opencv installed using `apt-get`

In the build directory, run the below command --

```bash
cmake -DBUILD_CAFFE=ON ..
```

#### SCENARIO 2 -- Caffe and OpenCV already installed

In this example, we assume that Caffe and OpenCV are already present. The user needs to supply the paths of the library
to CMake. For OpenCV, specify the `OpenCV_DIR` which is where the user built OpenCV. For Caffe, specify the includes
directory and library using the `Caffe_INCLUDE_DIRS` and `Caffe_LIBS` variables. This will be where you installed Caffe.
Below is an example of the same.

```bash
cmake -DOpenCV_DIR=/home/"${USER}"/softwares/opencv/build \
  -DCaffe_INCLUDE_DIRS=/home/"${USER}"/softwares/caffe/build/install/include \
  -DCaffe_LIBS=/home/"${USER}"/softwares/caffe/build/install/lib/libcaffe.so ..
```

#### SCENARIO 3 -- OpenCV already installed

If Caffe is not already present but OpenCV is, then use the below command.

```bash
cmake -DOpenCV_DIR=/home/"${USER}"/softwares/opencv/build -DBUILD_CAFFE=ON
```

### Build the library 

CMake has created the makefiles for us and the next step is to build the project. Make sure that you are in the `build`
directory of the project and run the below 2 commands.

```
no_cores=`cat /proc/cpuinfo | grep processor | wc -l`
make -j${cores}
```

# Run the OpenPose

Make sure that you are in the root directory of the project. Run the OpenPose demo using --

```
./build/examples/openposeose/openpose.bin
```
