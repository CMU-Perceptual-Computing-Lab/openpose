#!/bin/bash

### INSTALL PREREQUISITES

UBUNTU_VERSION="$(lsb_release -r)"

# Basic
sudo apt-get -yq update
sudo apt-get -yq install build-essential
# General dependencies
sudo apt-get -yq install libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get -yq install --no-install-recommends libboost-all-dev
# Remaining dependencies, 14.04
sudo apt-get -yq install libgflags-dev libgoogle-glog-dev liblmdb-dev
if [[ ! $WITH_PYTHON == false ]]
then
    # Python2 libs (Official Ubuntu support dropped after Ubuntu 20)
    if [[ $UBUNTU_VERSION == *"14."* ]] || [[ $UBUNTU_VERSION == *"15."* ]] || [[ $UBUNTU_VERSION == *"16."* ]] || [[ $UBUNTU_VERSION == *"17."* ]] || [[ $UBUNTU_VERSION == *"18."* ]];
    then
        sudo apt-get -yq install python-setuptools python-dev build-essential
        hash pip2 2> /dev/null || sudo apt-get -yq install python-pip
        sudo -H pip2 install pip --upgrade
        if [[ "$CI" == "true" ]]
        then
            sudo -H pip2 install --upgrade "numpy<1.17" protobuf
            sudo -H pip2 install --user "opencv-python<4.3"
        else
            sudo -H pip2 install --upgrade "numpy<1.17" protobuf "opencv-python<4.3"
        fi
    else
    # Python3 libs
        sudo apt-get -yq install python3-setuptools python3-dev build-essential
        hash pip3 2> /dev/null || sudo apt-get -yq install python3-pip
        sudo -H pip3 install pip --upgrade
        if [[ "$CI" == "true" ]]
        then
            sudo -H pip3 install --upgrade numpy protobuf
            sudo -H pip3 install --user opencv-python
        else
            sudo -H pip3 install --upgrade numpy protobuf opencv-python
        fi
    fi
fi
# OpenCL Generic (Official OpenPose support dropped after Ubuntu 20)
if [[ $UBUNTU_VERSION == *"14."* ]] || [[ $UBUNTU_VERSION == *"15."* ]] || [[ $UBUNTU_VERSION == *"16."* ]] || [[ $UBUNTU_VERSION == *"17."* ]] || [[ $UBUNTU_VERSION == *"18."* ]]
then
    sudo apt-get -yq install opencl-headers ocl-icd-opencl-dev
    sudo apt-get -yq install libviennacl-dev
fi
# OpenCV 2.4 -> Added as option
# # sudo apt-get --assume-yes install libopencv-dev
