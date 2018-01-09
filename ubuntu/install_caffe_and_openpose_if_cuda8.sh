#!/bin/bash



echo "------------------------- Installing Caffe and OpenPose -------------------------"
echo "NOTE: This script assumes that CUDA and cuDNN are already installed on your machine. Otherwise, it might fail."



function exitIfError {
    if [[ $? -ne 0 ]] ; then
        echo ""
        echo "------------------------- -------------------------"
        echo "Errors detected. Exiting script. The software might have not been successfully installed."
        echo "------------------------- -------------------------"
        exit 1
    fi
}



function executeShInItsFolder {
    # $1 = sh file name
    # $2 = folder where the sh file is
    # $3 = folder to go back
    cd $2   
    exitIfError
    sudo chmod +x $1
    exitIfError
    bash ./$1
    exitIfError
    cd $3
    exitIfError
}



git submodule update --init --recursive
executeShInItsFolder "install_caffe_if_cuda8.sh" "./3rdparty/caffe" "../.."
exitIfError



executeShInItsFolder "install_openpose_if_cuda8.sh" "./ubuntu" "./"
exitIfError



echo "------------------------- Caffe and OpenPose Installed -------------------------"
echo ""
