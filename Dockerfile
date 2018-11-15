FROM ubuntu:16.04
MAINTAINER Vlad Klim 'vladsklim@gmail.com'

# Установка необходимых пакетов для Ubuntu
RUN apt-get update -y
RUN apt-get install -y build-essential python3.5 python3.5-dev python3-pip python3-setuptools python3-tk make git scons gcc pkg-config pulseaudio libpulse-dev portaudio19-dev libglibmm-2.4-dev libasound-dev libao4 libao-dev sonic sox swig flite1-dev ffmpeg libav-tools x264 x265 locales locales-all alsa-utils dbus-x11 net-tools
RUN ldconfig

# Запуск D-bus
RUN export $(dbus-launch)

# Копирование файлов проекта
COPY . /work
WORKDIR /work

# Установка модулей для Python3
RUN pip3 install --upgrade pip
RUN pip3 install decorator flask==1.0.2 flask-httpauth==3.2.4 gensim gevent==1.3.7 h5py tensorflow keras matplotlib numpy pocketsphinx pydub simpleaudio requests git+https://github.com/datalogai/recurrentshop.git git+https://github.com/farizrahman4u/seq2seq.git
RUN ldconfig

# Установка CMUclmtk_v0.7
WORKDIR /work/cmuclmtk-0.7
RUN ./configure
RUN make install
WORKDIR /work

# Установка RHVoice
WORKDIR /work/RHVoice
RUN scons --config=force
RUN scons install
WORKDIR /work

RUN ldconfig

# Копирование статической языковой модели, словаря и акустической модели для PocketSphinx
WORKDIR /work/Voice_ChatBot/temp
RUN cp prepared_questions.lm /usr/local/lib/python3.5/dist-packages/pocketsphinx/model/ru_bot.lm
RUN cp prepared_questions.dic /usr/local/lib/python3.5/dist-packages/pocketsphinx/model/ru_bot.dic
WORKDIR /work/Voice_ChatBot

WORKDIR /work
RUN cp -r zero_ru.cd_cont_4000 /usr/local/lib/python3.5/dist-packages/pocketsphinx/model/zero_ru.cd_cont_4000

# Удаление больше не нужных папок
RUN rm -rf cmuclmtk-0.7
RUN rm -rf RHVoice
RUN rm -rf zero_ru.cd_cont_4000

WORKDIR /work/Voice_ChatBot

# Изменение локализации для вывода кириллицы в терминале
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Очистка кеша
RUN apt-get -y autoremove
RUN apt-get -y autoclean
RUN apt-get -y clean

#ENTRYPOINT ["python3"]
#CMD ["rest_server.py"]
