OpenPose Doc - Python API
====================================

## Contents
1. [Introduction](#introduction)
2. [Advance Introduction (Optional)](#advance-introduction-optional)
3. [Compatibility](#compatibility)
4. [Installation](#installation)
5. [Testing And Developing](#testing-and-developing)
6. [Exporting Python OpenPose](#exporting-python-openpose)
7. [Common Issues](#common-issues)



## Introduction
Almost all the OpenPose functionality, but in Python!

When should you look at the [Python](03_python_api.md) or [C++](04_cpp_api.md) APIs? If you want to read a specific input, and/or add your custom post-processing function, and/or implement your own display/saving.

You should be familiar with the [**OpenPose Demo**](01_demo.md) and the main OpenPose flags before trying to read the C++ or Python API examples. Otherwise, it will be way harder to follow.



## Advance Introduction (Optional)
This module exposes a Python API for OpenPose. It is effectively a wrapper that replicates most of the functionality of the [op::Wrapper class](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/wrapper/wrapper.hpp) and allows you to populate and retrieve data from the [op::Datum class](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/core/datum.hpp) using standard Python and Numpy constructs.

The Python API is analogous to the C++ function calls. You may find them in [python/openpose/openpose_python.cpp#L194](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/python/openpose/openpose_python.cpp#L194).

The Python API is rather simple: `op::Array<float>` and `cv::Mat` objects get casted to numpy arrays automatically. Every other data structure based on the standard library is automatically converted into Python objects. For example, an `std::vector<std::vector<float>>` would become `[[item, item], [item, item]]`, etc. We also provide a casting of `op::Rectangle` and `op::Point` which simply expose setter getter for [x, y, width, height], etc.





## Compatibility
The OpenPose Python module is compatible with both Python 2 and Python 3 (default and recommended). In addition, it will also run in all OpenPose compatible operating systems. It uses [Pybind11](https://github.com/pybind/pybind11) for mapping between C++ and Python datatypes.

To compile, enable `BUILD_PYTHON` in CMake-gui, or run `cmake -DBUILD_PYTHON=ON ..` from your build directory. In Windows, make sure you compile the whole solution (clicking the green play button does not compile the whole solution!). You can do that by right-click on the OpenPose project solution, and clicking in `Build Solution` (or individually building the PyOpenPose module).

Pybind selects the latest version of Python by default (Python 3). To use Python 2, change `PYTHON_EXECUTABLE` and `PYTHON_LIBRARY` flags in CMake-gui to your desired Python version.

```
# Ubuntu
PYTHON_EXECUTABLE=/usr/bin/python2.7
PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7m.so
```

```
# Mac OSX
PYTHON_EXECUTABLE=/usr/local/bin/python2.7
PYTHON_LIBRARY=/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7m.dylib
```

```
:: Windows
PYTHON_EXECUTABLE=C:/Users/user/AppData/Local/Programs/Python/Python27/python.exe
```

If run via the command line, you may need to run cmake twice in order for this change to take effect.



## Installation
Make sure you followed the Python steps in [doc/installation/0_index.md#cmake-configuration](installation/0_index.md#cmake-configuration).



## Testing And Developing
All the Python examples from the Tutorial API Python module can be found in `build/examples/tutorial_api_python` in your build folder. Navigate directly to this path to run examples.

```
# From command line
cd build/examples/tutorial_api_python

# Python 3 (default version)
python3 01_body_from_image.py
python3 02_whole_body_from_image.py
# python3 [any_other_python_example.py]

# Python 2
python2 01_body_from_image.py
python2 02_whole_body_from_image.py
# python2 [any_other_python_example.py]
```

For quick prototyping, you can simply duplicate and rename any of the existing sample files in `build/examples/tutorial_api_python` within that same folder and start building in there. These files are copied from [existing example files](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python) on compiling time. 2 alternatives:
- You can either duplicate and create your files in [examples/tutorial_api_python/](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/tutorial_api_python), but you will have to recompile OpenPose every time you make changes to your Python files so they are copied over the `build/` folder.
- Or you can directly edit them in `build/examples/tutorial_api_python`. This does not require rebuilding, but cleaning OpenPose will remove the whole `build/` folder, so make sure to back your files up!



## Exporting Python OpenPose
Note: This step is only required if you are moving the `*.py` files outside their original location, or writing new `*.py` scripts outside `build/examples/tutorial_api_python`.

Ubuntu/OSX:

- Option a, installing OpenPose: On an Ubuntu or OSX based system, you could install OpenPose by running `sudo make install`, you could then set the OpenPose path in your python scripts to the OpenPose installation path (default: `/usr/local/python`) and start using OpenPose at any location. Take a look at `build/examples/tutorial_api_python/01_body_from_image.py` for an example.
- Option b, not installing OpenPose: To move the OpenPose Python API demos to a different folder, ensure that the line `sys.path.append('{OpenPose_path}/python')` is properly set in your `*.py` files, where `{OpenPose_path}` points to your build folder of OpenPose. Take a look at `build/examples/tutorial_api_python/01_body_from_image.py` for an example.

Windows:

- Ensure that the folder  `build/x{86/64}/Release`and `build/bin` are copied along with `build/python` As noted in the example, the path for these can be changed in the following two variables:

  ```
  sys.path.append(dir_path + '/../../python/openpose/Release);
  os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../{x86/x64}/Release;' +  dir_path + '/../../bin;'
  ```



## Common Issues
### Do not use PIL
In order to read images in Python, make sure to use OpenCV (do not use PIL). We found that feeding a PIL image format to OpenPose results in the input image appearing in grey and duplicated 9 times (so the output skeleton appear 3 times smaller than they should be, and duplicated 9 times).


### Cannot Import Name PyOpenPose
The error in general is that PyOpenPose cannot be found (an error similar to: `ImportError: cannot import name pyopenpose`). Ensure first that `BUILD_PYTHON` flag is set to ON. If the error persists, check the following:

In the script you are running, check for the following line, and run the following command in the same location as where the file is

**Ubuntu/OSX:**

`sys.path.append('../../python');`

```
ls ../../python/openpose
```

Check the contents of this location. It should contain one of the following files:

```
pyopenpose.cpython-35m-x86_64-linux-gnu.so
pyopenpose.so
```

If you do not have any one of those, you may not have compiled openpose successfully, or you may be running the examples, not from the build folder but the source folder. If you have the first one, you have compiled PyOpenPose for Python 3, and have to run the scripts with `python3`, and vice versa for the 2nd one. Follow the testing examples above for exact commands.

**Windows:**

Problem 1: If you are in Windows, and you fail to install the required third party Python libraries, it might print an error similar to: `Exception: Error: OpenPose library could not be found. Did you enable BUILD_PYTHON in CMake and have this Python script in the right folder?`. From GitHub issue #941:
```
I had a similar issue with Visual Studio (VS). I am pretty sure that the issue is that while you are compiling OpenPose in VS, it tries to import cv2 (python-opencv) and it fails. So make sure that if you open cmd.exe and run Python, you can actually import cv2 without errors. I could not, but I had cv2 installed in a IPython environment (Anaconda), so I activated that environment, and then ran (change this to adapt it to your VS version and location of OpenPose.sln):

C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild.exe C:\path\to\OpenPose.sln
```

Problem 2: Python for Openpose needs to be compiled in Release mode for now. This can be done in [Visual Studio](https://cdn.stereolabs.com/docs/getting-started/images/release_mode.png). Once that is done check this line:

`sys.path.append(dir_path + '/../../python/openpose/Release');`

```
dir ../../python/openpose/Release
```

Check the contents of this location. It should contain one of the following files:

```
pyopenpose.cp36-win_amd64.pyd
pyopenpose.pyd
```

If such a folder does not exist, you need to compile in Release mode as seen above. If you have the first one, you have compiled PyOpenPose for Python 3, and have to run the scripts with `python3`, and vice versa for the 2nd one. Follow the testing examples above for exact commands. If that still does not work, check this line:

`os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'`

```
dir ../../x64/Release
dir ../../bin
```

Ensure that both of these paths exist, as PyOpenPose needs to reference those libraries. If they don't exist, change the path so that they point to the correct location in your build folder.
