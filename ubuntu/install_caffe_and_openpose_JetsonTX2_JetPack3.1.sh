#!/bin/bash



echo "------------------------- Installing Caffe and OpenPose -------------------------"
echo "NOTE: This script assumes that just flashed JetPack 3.1 : Ubuntu 16, CUDA 8, cuDNN 6 and OpenCV are already installed on your machine. Otherwise, it might fail."



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
executeShInItsFolder "install_caffe_JetsonTX2_JetPack3.1.sh" "./3rdparty/caffe" "../.."
exitIfError



executeShInItsFolder "install_openpose_JetsonTX2_JetPack3.1.sh" "./ubuntu" "./"
exitIfError



echo "------------------------- Caffe and OpenPose Installed -------------------------"
echo ""
