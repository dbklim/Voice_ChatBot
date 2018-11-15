#!/bin/sh

if ! [ -d log ]
then mkdir log
fi

time python3 -uB bot.py predict -ss -sr | tee log/bot.log
