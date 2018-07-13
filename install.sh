#!/bin/bash



echo "------------------------- Installing CMake -------------------------"

function exitIfError {
    if [[ $? -ne 0 ]] ; then
        echo ""
        echo "------------------------- -------------------------"
        echo "Errors detected. Exiting script. The software might have not been successfully installed."
        echo "------------------------- -------------------------"
        exit 1
    fi
}

bash ./ubuntu/install_cmake.sh
exitIfError

echo "------------------------- Installing OpenCV Python -------------------------"
pip install opencv-python
exitIfError

echo "------------------------- Making Caffe and OpenPose -------------------------"
mkdir build
pushd build/
cmake ..
exitIfError
make -j`nproc`
exitIfError
popd
