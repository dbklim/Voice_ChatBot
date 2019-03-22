FROM ubuntu:18.04
MAINTAINER Vlad Klim 'vladsklim@gmail.com'

# Установка необходимых пакетов для Ubuntu alsa-utils dbus-x11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y tzdata build-essential python3.6 python3.6-dev python3-pip python3-setuptools python3-tk make git scons gcc pkg-config pulseaudio libpulse-dev portaudio19-dev libglibmm-2.4-dev libasound-dev libao4 libao-dev sonic sox swig flite1-dev ffmpeg x264 x265 locales locales-all net-tools  zip unzip dbus-x11

# Установка часового пояса хост-машины
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Установка модулей для Python3
RUN pip3 install --upgrade pip
RUN pip3 install decorator flask==1.0.2 flask-httpauth==3.2.4 gensim gevent==1.3.7 h5py tensorflow keras matplotlib numpy pocketsphinx pydub simpleaudio requests git+https://github.com/datalogai/recurrentshop.git git+https://github.com/Desklop/seq2seq.git

# Копирование файлов проекта
COPY . /Voice_ChatBot
WORKDIR /Voice_ChatBot

# Установка CMUclmtk_v0.7
RUN unzip install_files/cmuclmtk-0.7.zip
WORKDIR /Voice_ChatBot/cmuclmtk-0.7
RUN ./configure
RUN make
RUN make install
WORKDIR /Voice_ChatBot

# Установка RHVoice
RUN unzip install_files/RHVoice.zip
WORKDIR /Voice_ChatBot/RHVoice
RUN scons --config=force
RUN scons install
WORKDIR /Voice_ChatBot

RUN rm -rf cmuclmtk-0.7
RUN rm -rf RHVoice
RUN rm -rf install_files
RUN ldconfig

# Копирование статической языковой модели, словаря и акустической модели для PocketSphinx
WORKDIR /Voice_ChatBot/temp
RUN cp prepared_questions_plays_ru.lm /usr/local/lib/python3.6/dist-packages/pocketsphinx/model/ru_bot_plays_ru.lm
RUN cp prepared_questions_plays_ru.dic /usr/local/lib/python3.6/dist-packages/pocketsphinx/model/ru_bot_plays_ru.dic
RUN cp -r zero_ru.cd_cont_4000 /usr/local/lib/python3.6/dist-packages/pocketsphinx/model/zero_ru.cd_cont_4000
RUN rm -rf zero_ru.cd_cont_4000
WORKDIR /Voice_ChatBot

# Изменение локализации для вывода кириллицы в терминале
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Очистка кеша
RUN apt-get -y autoremove
RUN apt-get -y autoclean
RUN apt-get -y clean

ENTRYPOINT ./run_rest_server.sh
#CMD ["rest_server.py"]
