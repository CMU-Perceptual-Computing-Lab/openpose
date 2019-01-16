#!/bin/bash



echo "------------------------- Installing OpenPose -------------------------"



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



executeShInItsFolder "scripts/ubuntu_deprecated/copy_makefile_files.sh" "./" "./"



echo "------------------------- Compiling OpenPose -------------------------"
make all -j`nproc`
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
