#!/bin/bash

echo "This script assumes Ubuntu 16 or 14 and Nvidia Graphics card up to 10XX. Otherwise, it will fail."

if [[ $CI == true ]]; then
    WGET_VERBOSE="--no-verbose"
fi

# Install cuDNN 5.1
UBUNTU_VERSION="$(lsb_release -r)"
if [[ $UBUNTU_VERSION == *"14."* ]] || [[ $UBUNTU_VERSION == *"15."* ]] || [[ $UBUNTU_VERSION == *"16."* ]]; then
    CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
    echo "wget -c ${CUDNN_URL} ${WGET_VERBOSE}"
    wget -c ${CUDNN_URL} ${WGET_VERBOSE}
    sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
    rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
else
    echo "cuDNN NOT INSTALLED! Ubuntu 16 or 14 not found. Install cuDNN manually from 'https://developer.nvidia.com/cudnn'."
fi
