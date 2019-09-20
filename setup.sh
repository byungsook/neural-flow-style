#!/bin/bash

# # install virtualenv if needed
# pip3 install virtualenv
# virtualenv --system-site-packages ./venv
# source ./venv/bin/activate  # sh, bash, ksh, or zsh
# pip install --upgrade pip

# or run
sudo apt-get install cmake g++ git python3-dev qt5-qmake qt5-default

# install packages
pip3 install --user --upgrade tensorflow-gpu==1.12 tqdm matplotlib Pillow imageio scipy scikit-image

# install mantaflow
rm -rf manta/build
cd manta
mkdir build
cd build

cmake .. -DNUMPY=ON -DOPENMP=ON -DPYTHON_VERSION=3.6
make -j8