#!/bin/bash
# Начальное обновление системы
apt-get -y update
apt-get -y --force-yes dist-upgrade
# Список репозиториев
add-apt-repository ppa:jonathonf/ffmpeg-3
# Обновление информации о пакетах
apt-get -y update

PACKAGES="ffmpeg libav-tools x264 x265 python3.5 python3.5-dev python3-pip python3-tk python3-setuptools make git scons gcc pkg-config pulseaudio libpulse-dev portaudio19-dev libglibmm-2.4-dev libasound-dev libao4 libao-dev sonic sox swig flite1-dev net-tools dbus-x11"
# Установка пакетов Ubuntu
apt-get -y --force-yes install $PACKAGES
ldconfig

PACKAGES="decorator flask==1.0.2 flask-httpauth==3.2.4 gensim gevent==1.3.7 h5py keras matplotlib numpy pocketsphinx pydub simpleaudio requests git+https://github.com/datalogai/recurrentshop.git git+https://github.com/farizrahman4u/seq2seq.git"
# Установка пакетов Python3
yes | pip3 install --upgrade pip
yes | pip3 install $PACKAGES

# Установка CMUclmtk_v0.7
wget https://netcologne.dl.sourceforge.net/project/cmusphinx/cmuclmtk/0.7/cmuclmtk-0.7.tar.gz -O - | tar -xz
cd cmuclmtk-0.7
./configure
make install
cd -
rm -rf cmuclmtk-0.7

# Установка RHVoice
git clone https://github.com/Olga-Yakovleva/RHVoice.git
cd RHVoice
scons --config=force
scons install
cd -
rm -rf RHVoice
ldconfig
# Если не найден RHVoice-client - см. "Install RHVoice.txt"

if [[ $1 = 'gpu' ]]
then 
    yes | pip3 install tensorflow-gpu==1.12.0
    # install CUDA Toolkit v9.0
    # instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb)
    CUDA_REPO_PKG="cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb"
    wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${CUDA_REPO_PKG}
    sudo dpkg -i ${CUDA_REPO_PKG}
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda-9-0
    
    CUDA_PATCH1="cuda-repo-ubuntu1604-9-0-local-cublas-performance-update_1.0-1_amd64-deb"
    wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/${CUDA_PATCH1}
    sudo dpkg -i ${CUDA_PATCH1}
    sudo apt-get update
    
    # install cuDNN v7.0
    CUDNN_PKG="libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb"
    wget https://github.com/ashokpant/cudnn_archive/raw/master/v7.0/${CUDNN_PKG}
    sudo dpkg -i ${CUDNN_PKG}
    sudo apt-get update
    
    # install NVIDIA CUDA Profile Tools Interface (libcupti-dev v9.0)
    sudo apt-get install cuda-command-line-tools-9-0
    
    # set environment variables
    export PATH=${PATH}:/usr/local/cuda-9.0/bin
    export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
else 
    yes | pip3 install tensorflow==1.12.0
fi
