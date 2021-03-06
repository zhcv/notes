#!/usr/bin/env bash

<<!
# relay on packages below.
sudo apt-get install build-essential cmake pkg-config python2.7-dev python3.5-dev \
  libprotobuf-dev protobuf-compiler libatlas-base-dev gfortran \
  libgtk-3-dev libgphoto2-dev libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
!

sudo pip3 install -U numpy

wget -O opencv-3.3.0.zip https://github.com/opencv/opencv/archive/3.3.0.zip

wget -O opencv_contrib-3.3.0.zip https://github.com/opencv/opencv_contrib/archive/3.3.0.zip

wget https://github.com/opencv/opencv_3rdparty/raw/ippicv/master_20170418/ippicv/ippicv_2017u2_lnx_intel64_20170418.tgz

wget -O tiny-dnn-1.0.0a3.tar.gz https://github.com/tiny-dnn/tiny-dnn/archive/v1.0.0a3.tar.gz

unzip opencv-3.3.0.zip

unzip opencv_contrib-3.3.0.zip

mkdir -p opencv-3.3.0/.cache/ippicv

cp ippicv_2017u2_lnx_intel64_20170418.tgz \
  opencv-3.3.0/.cache/ippicv/87cbdeb627415d8e4bc811156289fa3a-ippicv_2017u2_lnx_intel64_20170418.tgz

mkdir -p opencv-3.3.0/.cache/tiny_dnn

cp tiny-dnn-1.0.0a3.tar.gz opencv-3.3.0/.cache/tiny_dnn/adb1c512e09ca2c7a6faef36f9c53e59-v1.0.0a3.tar.gz

mkdir -p opencv-3.3.0/build

cd opencv-3.3.0/build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_CUDA=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
  -D OPENCV_EXTRA_MODULES_PATH=/full/path/to/opencv_contrib-3.3.0/modules \
  ..

make "-j$(nproc)"

sudo make install

sudo ldconfig

ls -l /usr/local/lib/python3.5/dist-packages/

sudo mv /usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.5/dist-packages/cv2.so

sudo pip3 install -U opencv-python opencv-contrib-python

python3 -c 'import cv2; print(cv2.__version__)'

ls /usr/local/share/OpenCV/java

sudo sh -c 'echo "/usr/local/share/OpenCV/java" > /etc/ld.so.conf.d/opencv.conf'

sudo ldconfig


