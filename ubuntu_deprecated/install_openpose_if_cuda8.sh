#!/bin/bash



echo "------------------------- Installing OpenPose -------------------------"
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
    ./$1
    exitIfError
    cd $3
    exitIfError
}



echo "------------------------- Checking Ubuntu Version -------------------------"
ubuntu_version="$(lsb_release -r)"
echo "Ubuntu $ubuntu_version"
if [[ $ubuntu_version == *"14."* ]]; then
    ubuntu_le_14=true
elif [[ $ubuntu_version == *"16."* || $ubuntu_version == *"15."* || $ubuntu_version == *"17."* || $ubuntu_version == *"18."* ]]; then
    ubuntu_le_14=false
else
    echo "Ubuntu release older than version 14. This installation script might fail."
    ubuntu_le_14=true
fi
exitIfError
echo "------------------------- Ubuntu Version Checked -------------------------"
echo ""



echo "------------------------- Checking Number of Processors -------------------------"
NUM_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
echo "$NUM_CORES cores"
exitIfError
echo "------------------------- Number of Processors Checked -------------------------"
echo ""



echo "------------------------- Compiling OpenPose -------------------------"
# Go back to main folder
cd ..
# Copy Makefile & Makefile.config
cp ubuntu/Makefile.example Makefile
if [[ $ubuntu_le_14 == true ]]; then
    cp ubuntu/Makefile.config.Ubuntu14_cuda8.example Makefile.config
else
    cp ubuntu/Makefile.config.Ubuntu16_cuda8.example Makefile.config
fi
# Compile OpenPose
make all -j$NUM_CORES
exitIfError
echo "------------------------- OpenPose Compiled -------------------------"
echo ""



echo "------------------------- Downloading OpenPose Models -------------------------"
executeShInItsFolder "getModels.sh" "./models" ".."
exitIfError
echo "Models downloaded"
echo "------------------------- OpenPose Models Downloaded -------------------------"
echo ""



echo "------------------------- OpenPose Installed -------------------------"
echo ""
