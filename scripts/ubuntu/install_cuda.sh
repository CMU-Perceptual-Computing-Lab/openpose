#!/bin/bash

echo "NOTE: This script assumes Ubuntu 20 or 18 (Nvidia Graphics card >= 10XX), as well as 16 or 14 (card up to 10XX)."
echo "Otherwise, install it by yourself or it might fail."

if [[ $CI == true ]]; then
    WGET_VERBOSE="--no-verbose"
fi

# Install CUDA 8.0
UBUNTU_VERSION="$(lsb_release -r)"
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
if [[ $UBUNTU_VERSION == *"14."* ]]; then
    CUDA_LINK=https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
    echo "wget -c \"$CUDA_LINK\" ${WGET_VERBOSE}"
    wget -c "$CUDA_LINK" ${WGET_VERBOSE}
    sudo dpkg --install cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo apt-get update
    sudo apt-get install cuda-8-0
elif [[ $UBUNTU_VERSION == *"16."* ]]; then
    CUDA_LINK=https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    echo "wget -c \"$CUDA_LINK\" ${WGET_VERBOSE}"
    wget -c "$CUDA_LINK" ${WGET_VERBOSE}
    sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo apt-get update
    sudo apt-get install cuda-8-0
# Install CUDA 10.0
elif [[ $UBUNTU_VERSION == *"18."* ]]; then
    CUDA_PIN_LINK=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    CUDA_LINK=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    echo "wget -c \"$CUDA_PIN_LINK\" ${WGET_VERBOSE}"
    wget -c "$CUDA_PIN_LINK" ${WGET_VERBOSE}
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    echo "wget \"$CUDA_LINK\" ${WGET_VERBOSE}"
    wget "$CUDA_LINK" ${WGET_VERBOSE}
    sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda
# Install CUDA 11.0
elif [[ $UBUNTU_VERSION == *"20."* ]]; then
    CUDA_PIN_LINK=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    CUDA_LINK=https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
    echo "wget -c \"$CUDA_PIN_LINK\" ${WGET_VERBOSE}"
    wget -c "$CUDA_PIN_LINK" ${WGET_VERBOSE}
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    echo "wget \"$CUDA_LINK\" ${WGET_VERBOSE}"
    wget "$CUDA_LINK" ${WGET_VERBOSE}
    sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
    sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda
fi
# sudo apt-get install cuda
