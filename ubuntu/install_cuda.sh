ubuntu_version="$(lsb_release -r)"
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
if [[ $ubuntu_version == *"14."* ]]; then
  wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb"
  sudo dpkg --install cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
elif [[ $ubuntu_version == *"16."* ]]; then
  wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb"
  sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
fi
sudo apt-get update
sudo apt-get install cuda
