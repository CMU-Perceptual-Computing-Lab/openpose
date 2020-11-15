#!/bin/bash

echo "NOTE: This script assumes Ubuntu 20 or 18 (Nvidia Graphics card >= 10XX), as well as 16 or 14 (card up to 10XX)."
echo "Otherwise, install it by yourself or it might fail."

# Install CUDA 8.0
ubuntu_version="$(lsb_release -r)"
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
if [[ $ubuntu_version == *"14."* ]]; then
  wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb"
  sudo dpkg --install cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
  sudo apt-get update
  sudo apt-get install cuda-8-0
elif [[ $ubuntu_version == *"16."* ]]; then
  wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb"
  sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
  sudo apt-get update
  sudo apt-get install cuda-8-0
# Install CUDA 10.0
elif [[ $ubuntu_version == *"18."* ]]; then
  wget -c "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin"
  sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
  sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
  sudo apt-get update
  sudo apt-get -y install cuda
# Install CUDA 11.0
elif [[ $ubuntu_version == *"20."* ]]; then
  wget -c "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"
  sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget "https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb"
  sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
  sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
  sudo apt-get update
  sudo apt-get -y install cuda
fi
# sudo apt-get install cuda
