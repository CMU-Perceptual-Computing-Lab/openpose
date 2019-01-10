#!/bin/bash



echo "------------------------- Preprocessing -------------------------"
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



echo "------------------------- Copying Required Files -------------------------"
# Copy Makefile & Makefile.config
cp scripts/ubuntu/Makefile.example Makefile
if [[ $ubuntu_le_14 == true ]]; then
    cp scripts/ubuntu_deprecated/Makefile.config.Ubuntu14_cuda8.example Makefile.config
else
    cp scripts/ubuntu_deprecated/Makefile.config.Ubuntu16_cuda8.example Makefile.config
fi
exitIfError
echo "------------------------- Required Files Copied -------------------------"



echo "------------------------- Preprocessing Ready -------------------------"
echo ""
