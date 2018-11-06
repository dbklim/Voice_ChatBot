#!/bin/bash
# Начальное обновление системы
apt-get -y update
apt-get -y --force-yes dist-upgrade
# Список репозиториев
add-apt-repository ppa:jonathonf/ffmpeg-3
# Обновление информации о пакетах
apt-get -y update

PACKAGES="ffmpeg libav-tools x264 x265 python3.5 python3-pip scons gcc pkg-config pulseaudio libpulse-dev portaudio19-dev libglibmm-2.4-dev libasound-dev sonic sox swig"
# Установка пакетов Ubuntu
apt-get -y --force-yes install $PACKAGES
ldconfig

PACKAGES="decorator flask==1.0.2 flask-httpauth==3.2.4 gensim gevent==1.3.7 h5py keras matplotlib numpy pocketsphinx pydub requests git+https://github.com/datalogai/recurrentshop.git git+https://github.com/farizrahman4u/seq2seq.git"
# Установка пакетов Python3
/usr/bin/yes | pip3 install $PACKAGES

# Установка CMUclmtk
git clone https://github.com/skerit/cmusphinx.git
cd cmusphinx/cmuclmtk
make install
rm -rf cmusphinx

# Установка RHVoice
git clone https://github.com/Olga-Yakovleva/RHVoice.git
cd RHVoice
scons --config=force
scons install
rm -rf RHVoice
ldconfig
