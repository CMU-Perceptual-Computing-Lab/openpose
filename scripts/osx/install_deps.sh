#!/bin/bash
brew_packages="openblas snappy leveldb gflags glog szip lmdb hdf5 opencv protobuf boost cmake viennacl"
for pkg in $brew_packages
do
    echo "brew install $pkg || brew upgrade $pkg"
    brew install "$pkg" || brew upgrade "$pkg"
done

# with Python pycaffe needs dependencies built from source
#brew install --build-from-source --with-python -vd protobuf
#brew install --build-from-source -vd boost boost-python
# without Python the usual installation suffices

pip_packages="numpy<1.17 opencv-python<4.3"
for pkg in $pip_packages
do
    echo "sudo -H python2 -m pip install $pkg"
    sudo -H python2 -m pip install "$pkg"
done
