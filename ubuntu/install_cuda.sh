sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
if [[ $ubuntu_version == *"14."* ]]; then
    wget -c "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
    sudo dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
elif [[ $ubuntu_version == *"16."* ]]; then
    wget -c "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
if
sudo apt-get update
sudo apt-get install cuda
