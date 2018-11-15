#!/bin/sh

if ! [ -d log ]
then mkdir log
fi

time python3 -uB bot.py train | tee log/bot.log
