#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Синтез речи с помощью RHVoice.
'''

import subprocess
import os
from pydub import AudioSegment
from pydub.playback import play

# Создание временной папки, если она была удалена
if not os.path.exists('temp'):
    os.makedirs('temp')


class TextToSpeech:
    ''' Предназначен для синтеза речи с помощью RHVoice.
    1. voice - имя голоса, возможные значения: aleksandr, anna, elena и irina '''
    def __init__(self, voice='anna'):        
        if voice == 'aleksandr':
            self.voice = voice + '+alan'
        elif voice == 'anna':
            self.voice = voice + '+slt'
        elif voice == 'elena':
            self.voice = voice + '+slt'
        elif voice == 'irina':
            self.voice = voice + '+clb'
        else:
            print('\n[E] Неподдерживаемый голос, возможные варианты: aleksandr, anna, elena, irina\n')
            self.voice = 'anna+slt'


    def get(self, text, f_name_audio=None):
        ''' Синтез речи с помощью RHVoice.
        1. text - строка с текстом, который нужно синтезировать
        2. f_name_audio - имя .wav файла для сохранения синтезированной речи или None, что бы сразу же воспроизвести синтезированную речь 
        (при этом она будет временно сохранена в temp/synthesized_speech.wav) '''

        #if f_name_audio is None:
        #    f_name_audio = 'temp/answer.wav'
        
        # Запись синтезированной речи в .wav файл с частотой дискретизации 32 кГц и глубиной 16 бит, моно, используется sox
        command_line = "echo '" + text + "' | RHVoice-client -s " + self.voice + " "
        command_line += "| sox -t wav - -r 32000 -c 1 -b 16 -t wav - >'" + os.path.dirname(os.path.realpath(__file__)) + '/'
        if f_name_audio is None:
            command_line += "temp/synthesized_speech.wav'"
            subprocess.call(command_line, shell=True)
            sound = AudioSegment.from_wav(os.path.dirname(os.path.realpath(__file__)) + '/temp/synthesized_speech.wav')
            play(sound)
            # command_line += "| aplay" - не используется, т.к. при каждом обращении выводится сообщение от самого RHVoice в терминал
        else:
            command_line += f_name_audio + "'"
            subprocess.call(command_line, shell=True)
     
# Если вылазит FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe', нужно выполнить apt-get install ffmpeg libavcodec-extra


def main():
    tts = TextToSpeech('anna')
    tts.get('ты любишь меня?')


if __name__ == '__main__':
    main()
        

''' 
В данном варианте нужно как-то ждать завершения воспроизведения, т.к. выполнения основного потока не прерывается.
Тоже самое при использовании pygame.mixer
import pyglet
explosion = pyglet.media.load('/home/vladislav/Проекты/test.wav', streaming=False)
explosion.play()
pyglet.app.run() 
'''