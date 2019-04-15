#!/bin/bash

### INSTALL PREREQUISITES

sudo apt-get -y update

# Core
cat deps-core.txt | xargs sudo apt-get -y --no-install-recommends install
# General dependencies
cat deps-gen.txt | xargs sudo apt-get -y --no-install-recommends install
# Remaining dependencies, 14.04
cat deps-rem.txt | xargs sudo apt-get -y --no-install-recommends install
# Python2 libs
cat deps-python2.txt | xargs sudo apt-get -y --no-install-recommends install
sudo easy_install pip
sudo -H pip install --upgrade -r python-requirements.txt
# Python3 libs
cat deps-python3.txt | xargs sudo apt-get -y --no-install-recommends install
sudo -H pip3 install --upgrade -r python-requirements.txt
# OpenCV 2.4 -> Added as option
# # sudo apt-get -y install libopencv-dev
# OpenCL Generic
cat deps-opencl.txt | xargs sudo apt-get -y --no-install-recommends install
