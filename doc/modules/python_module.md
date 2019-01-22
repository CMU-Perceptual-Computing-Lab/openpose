# OpenPose Python Module and Demo

## Contents
1. [Introduction](#introduction)
2. [Compatibility](#compatibility)
3. [Installation](#installation)
4. [Testing](#testing)
5. [Exporting Python OpenPose](#exporting-python-openpose)



## Introduction
This module exposes a Python API for OpenPose. It is effectively a wrapper that replicates most of the functionality of the [op::Wrapper class](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/wrapper/wrapper.hpp) and allows you to populate and retrieve data from the [op::Datum class](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/core/datum.hpp) using standard Python and Numpy constructs.



## Compatibility
The OpenPose Python module is compatible with both Python 2 and Python 3. In addition, it will also run in all OpenPose compatible operating systems. It uses [Pybind11](https://github.com/pybind/pybind11) for mapping between C++ and Python datatypes.

To compile, enable `BUILD_PYTHON` in cmake. Pybind selects the latest version of Python by default (Python 3). To use Python 2, change `PYTHON_EXECUTABLE` and `PYTHON_LIBRARY` flags in cmake-gui to your desired python version. 

Ubuntu Eg:

```
PYTHON_EXECUTABLE=/usr/bin/python2.7 
PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7m.so
```

Mac OSX Eg:

```
PYTHON_EXECUTABLE=/usr/local/bin/python2.7 
PYTHON_LIBRARY=/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7m.dylib 
```

Windows Eg:

```
PYTHON_EXECUTABLE=C:/Users/user/AppData/Local/Programs/Python/Python27/python.exe
```

If run via the command line, you may need to run cmake twice in order for this change to take effect. 



## Installation
Check [doc/installation.md#python-module](../installation.md#python-api) for installation steps.

The Python API requires python-dev, Numpy (for array management), and OpenCV (for image loading). They can be installed via:

```
# Python 2
sudo apt-get install python-dev
sudo pip install numpy opencv-python
# Python 3 (recommended)
sudo apt-get install python3-dev
sudo pip3 install numpy opencv-python
```



## Testing
All the Python examples from the Tutorial API Python module can be found in `build/examples/tutorial_api_python` in your build folder. Navigate directly to this path to run examples.

```
# From command line
cd build/examples/tutorial_api_python

# Python 2
python2 1_body_from_image.py
python2 2_whole_body_from_image.py
# python2 [any_other_example.py]

# Python 3 (recommended)
python3 1_body_from_image.py
python3 2_whole_body_from_image.py
# python3 [any_other_example.py]
```



## Exporting Python OpenPose
Note: This step is only required if you are moving the `*.py` files outside their original location, or writting new `*.py` scripts outside `build/examples/tutorial_api_python`.

Ubuntu/OSX:

- Option a, installing OpenPose: On an Ubuntu or OSX based system, you could install OpenPose by running `sudo make install`, you could then set the OpenPose path in your python scripts to the OpenPose installation path (default: `/usr/local/python`) and start using OpenPose at any location. Take a look at `build/examples/tutorial_api_python/1_body_from_image.py` for an example.
- Option b, not installing OpenPose: To move the OpenPose Python API demos to a different folder, ensure that the line `sys.path.append('{OpenPose_path}/python')` is properly set in your `*.py` files, where `{OpenPose_path}` points to your build folder of OpenPose. Take a look at `build/examples/tutorial_api_python/1_body_from_image.py` for an example.

Windows:

- Ensure that the folder  `build/x{86/64}/Release`and `build/bin` are copied along with `build/python` As noted in the example, the path for these can be changed in the following two variables:

  ```
  sys.path.append(dir_path + '/../../python/openpose/Release);
  os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../{x86/x64}/Release;' +  dir_path + '/../../bin;'
  ```

  

#### Common Issues
The error in general is that openpose cannot be found. Ensure first that `BUILD_PYTHON` flag is set to ON. If the error persists, check the following:

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

If you do not have any one of those, you may not have compiled openpose successfully, or you may be running the examples, not from the build folder but the source folder. If you have the first one, you have compiled pyopenpose for python 3, and have to run the scripts with python3, and vice versa for the 2nd one. Follow the testing examples above for exact commands.

**Windows:**

Python for Openpose needs to be compiled in Release mode for now. This can be done in [Visual Studio](https://cdn.stereolabs.com/docs/getting-started/images/release_mode.png). Once that is done check this line:

`sys.path.append(dir_path + '/../../python/openpose/Release');`

```
dir ../../python/openpose/Release
```

Check the contents of this location. It should contain one of the following files:

```
pyopenpose.cp36-win_amd64.pyd
pyopenpose.pyd
```

If such a folder does not exist, you need to compile in Release mode as seen above. If you have the first one, you have compiled pyopenpose for python 3, and have to run the scripts with python3, and vice versa for the 2nd one. Follow the testing examples above for exact commands. If that still does not work, check this line:

`os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'`

```
dir ../../x64/Release
dir ../../bin
```

Ensure that both of these paths exist, as pyopenpose needs to reference those libraries. If they don't exist, change the path so that they point to the correct location in your build folder

