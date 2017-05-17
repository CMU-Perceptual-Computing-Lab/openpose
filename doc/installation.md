OpenPose Library - Compilation and Installation
====================================



## Requirements
- Windows (tested on Windows 10)
- GPU with at least 2 GB and 1.5 GB available.
- CUDA and cuDNN installed.
- At least 2 GB of free RAM memory.
- Highly recommended: A CPU with at least 8 cores.

Note: These requirements assume the default configuration (i.e. `--net_resolution "656x368"` and `num_scales 1`). You might need more (with a greater net resolution and/or number of scales) or less resources (with smaller net resolution and/or using the MPI and MPI_4 models).



## Installation (Only available release mode supported)
**Highly important**: This script only works with CUDA 8 and Visual Studio 2015. Otherwise, check [Manual Compilation](#manual-compilation) from master documentation.

1. Visual Studio 2015
2. Required: CUDA, cuDNN must be already installed on your machine.
3. Build Caffe and other dependencies: caffe, opencv, openblas, boost will be installed in the folder 3rdparty with the script "buil_win.cmd". For more information, check the urls:

https://github.com/BVLC/caffe/tree/windows
http://caffe.berkeleyvision.org/installation.html

Optional:
If python layer is enabled (script build_win.cmd: BUILD_PYTHON, BUILD_PYTHON_LAYER) install python or miniconda (tested in version 2.7). 
Instructions to setup miniconda:
##Add the required channels
	conda config --add channels conda-forge
    conda config --add channels willyd
##Update conda
    conda update conda -y
##Download other required packages
    conda install --yes cmake ninja numpy scipy protobuf six scikit-image pyyaml pydotplus graphviz
	
	
3. Open the solution Openpose.sln with visual studio and buil it.It contains the openpose library and the demo "Openpose"


### Manual Compilation


