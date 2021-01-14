brew install openblas
brew install -vd snappy leveldb gflags glog szip lmdb
brew install hdf5 opencv
# with Python pycaffe needs dependencies built from source
#brew install --build-from-source --with-python -vd protobuf
#brew install --build-from-source -vd boost boost-python
# without Python the usual installation suffices
brew install protobuf boost
brew install cmake
brew install viennacl
sudo python2 -m pip install "numpy<1.17"
sudo python2 -m pip install "opencv-python<4.3"
