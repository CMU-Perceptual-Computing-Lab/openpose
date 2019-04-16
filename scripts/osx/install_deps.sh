cat scripts/osx/deps-core.txt | xargs brew install -vd
# with Python pycaffe needs dependencies built from source
#brew install --build-from-source --with-python -vd protobuf
#brew install --build-from-source -vd boost boost-python
# without Python the usual installation suffices
cat scripts/osx/deps-rem.txt | xargs brew install -vd
sudo pip install -r scripts/osx/python-requirements.txt
