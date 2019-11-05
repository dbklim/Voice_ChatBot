#!/bin/bash

# Что бы преобразовать ссылку на файл с гугл диска для скачивания с помощью wget, нужно в https://drive.google.com/uc?export=download&id= добавить id нужного файла, полученный из ссылки на файл, которую даёт сам гугл диск. Например:
# Ссылка на файл, которую даёт гугл диск: https://drive.google.com/file/d/1_hUmsu1d2bigS2Lpkm0VRNiL7GFF_AvE/view
# Ссылка для wget: https://drive.google.com/uc?export=download&id=1_hUmsu1d2bigS2Lpkm0VRNiL7GFF_AvE

# Если предыдущий вариант не работает (скорее всего по причине того, что файл очень большой), нужно использовать: wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
# Где FILEID заменить на id файла (который следует после id= в ссылке на файл, которую даёт гугл диск), FILENAME - на имя, под которым будет сохранён скачаный файл

mkdir -p data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_hUmsu1d2bigS2Lpkm0VRNiL7GFF_AvE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_hUmsu1d2bigS2Lpkm0VRNiL7GFF_AvE" -O data/Voice_ChatBot_data_04.11.2019.zip
rm -f /tmp/cookies.txt
unzip -o data/Voice_ChatBot_data_04.11.2019.zip -d .
rm -f data/Voice_ChatBot_data_04.11.2019.zip README_data.md
