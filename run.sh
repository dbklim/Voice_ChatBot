#!/bin/sh

if ! [ -d log ]
then mkdir log
fi

time python3 -uB bot.py predict | tee log/bot.log
