#!/bin/bash
apt-get -y update
apt-get -y --force-yes dist-upgrade

PACKAGES="ffmpeg x264 x265 python3.6 python3.6-dev python3-pip python3-setuptools make git scons gcc pkg-config pulseaudio libpulse-dev portaudio19-dev libglibmm-2.4-dev libasound-dev libao4 libao-dev sonic sox swig flite1-dev net-tools zip unzip"
# Установка пакетов Ubuntu
apt-get -y --force-yes install $PACKAGES

PACKAGES="decorator==4.4.0 flask==1.0.2 flask-httpauth==3.2.4 gensim==3.7.1 gevent==1.3.7 h5py==2.9.0 keras==2.2.4 matplotlib==3.0.3 numpy==1.16.2 pocketsphinx==0.1.15 pydub==0.23.1 simpleaudio==1.0.2 requests==2.18.4 git+https://github.com/datalogai/recurrentshop.git git+https://github.com/Desklop/seq2seq.git"
# Установка пакетов Python3
yes | pip3 install --upgrade pip
yes | pip3 install $PACKAGES

# Установка CMUclmtk_v0.7
unzip install_files/cmuclmtk-0.7.zip
cd cmuclmtk-0.7
./configure
make
make install
cd -
rm -rf cmuclmtk-0.7

# Установка RHVoice
unzip install_files/RHVoice.zip
cd RHVoice
scons --config=force
scons install
cd -
rm -rf RHVoice
ldconfig
# Если не найден RHVoice-client - см. "install_files/Install RHVoice.txt"

# Копирование языковой модели, словаря и акустической модели для PocketSphinx
cd temp
cp prepared_questions_plays_ru.lm /usr/local/lib/python3.6/dist-packages/pocketsphinx/model/ru_bot_plays_ru.lm
cp prepared_questions_plays_ru.dic /usr/local/lib/python3.6/dist-packages/pocketsphinx/model/ru_bot_plays_ru.dic
cp -r zero_ru.cd_cont_4000 /usr/local/lib/python3.6/dist-packages/pocketsphinx/model/zero_ru.cd_cont_4000
cd -

# Загрузка обучающих данных и моделей из Google Drive
./install_files/download_train_data_and_models.sh

if [[ $1 = 'gpu' ]]
then 
    yes | pip3 install tensorflow-gpu==1.13.1
    ./install_files/download_CUDA10.0_cuDNN.sh
    ./install_files/Install_CUDA10.0_cuDNN/install.sh
    rm -rf install_files/Install_CUDA10.0_cuDNN
else 
    yes | pip3 install tensorflow==1.13.1
fi
