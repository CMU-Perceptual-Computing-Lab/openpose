#!/bin/bash

echo "NOTE: This script assumes Ubuntu 16 or 14 and Nvidia Graphics card up to 10XX. Otherwise, install it by yourself or it will fail."

# Install cuDNN 5.1
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget -c ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
