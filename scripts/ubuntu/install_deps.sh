#!/bin/bash

### INSTALL PREREQUISITES

sudo apt-get -y update

# Core
cat scripts/ubuntu/deps-core.txt | xargs sudo apt-get -y --no-install-recommends install
# General dependencies
cat scripts/ubuntu/deps-gen.txt | xargs sudo apt-get -y --no-install-recommends install
# Remaining dependencies, 14.04
cat scripts/ubuntu/deps-rem.txt | xargs sudo apt-get -y --no-install-recommends install
# Python2 libs
cat scripts/ubuntu/deps-python2.txt | xargs sudo apt-get -y --no-install-recommends install
sudo easy_install pip
sudo -H pip install --upgrade -r scripts/ubuntu/python-requirements.txt
# Python3 libs
cat scripts/ubuntu/deps-python3.txt | xargs sudo apt-get -y --no-install-recommends install
sudo -H pip3 install --upgrade -r scripts/ubuntu/python-requirements.txt
# OpenCV 2.4 -> Added as option
# # sudo apt-get -y install libopencv-dev
# OpenCL Generic
cat scripts/ubuntu/deps-opencl.txt | xargs sudo apt-get -y --no-install-recommends install
