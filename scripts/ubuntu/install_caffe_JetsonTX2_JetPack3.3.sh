#!/bin/bash



echo "------------------------- Installing Caffe -------------------------"
echo "NOTE: This script assumes that just flashed JetPack 3.3 : Ubuntu 16, CUDA 9, cuDNN 7 and OpenCV are already installed on your machine. Otherwise, it might fail."



function exitIfError {
    if [[ $? -ne 0 ]] ; then
        echo ""
        echo "------------------------- -------------------------"
        echo "Errors detected. Exiting script. The software might have not been successfully installed."
        echo "------------------------- -------------------------"
        exit 1
    fi
}



echo "------------------------- Checking Ubuntu Version -------------------------"
# If you respected the installation assumptions, nothing to do. 
echo "------------------------- Ubuntu Version Checked -------------------------"
echo ""



echo "------------------------- Installing some Caffe Dependencies -------------------------"
# Basic
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install build-essential
# General dependencies
sudo apt-get --assume-yes install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev
sudo apt-get --assume-yes install --no-install-recommends libboost-all-dev
sudo apt-get --assume-yes install libgflags-dev libgoogle-glog-dev liblmdb-dev
# Python libs
sudo apt-get --assume-yes install python-pip python-dev build-essential
sudo -H pip install --upgrade pip
sudo -H pip install --upgrade numpy protobuf
# OpenCV is provided for tegra in JetPack
exitIfError
echo "------------------------- Some Caffe Dependencies Installed -------------------------"
echo ""



echo "------------------------- Compiling Caffe -------------------------"
cp Makefile.config.Ubuntu16_cuda9_JetsonTX2_caffe Makefile.config
make all -j`nproc` && make distribute -j`nproc`
# make test -j`nproc`
# make runtest -j`nproc`
exitIfError
echo "------------------------- Caffe Compiled -------------------------"
echo ""



echo "------------------------- Caffe Installed -------------------------"
echo ""
