# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для преобразования текста в речь с помощью RHVoice.
'''

import subprocess
import os
import sys
from pydub import AudioSegment
from pydub.playback import play


def tts(text, mode, filename_audio = None):
    ''' Преобразование текста в речь с помощью RHVoice.
    1. text - строка, которую необходимо преобразовать
    2. mode - может иметь два значения: into_file и playback
    2.1. into_file - запись синтезированной речи в .wav файл с частотой дискретизации 32 кГц и глубиной 16 бит, моно
    2.2. playback - воспроизведение речи сразу после синтезирования
    3. filename_audio - имя .wav файла для сохранения синтезированной речи в режиме into_file (если не задан, используется temp/answer.wav) '''

    if filename_audio == None:
        filename_audio = 'temp/answer.wav'
    # Запись синтезированной речи в .wav файл с частотой дискретизации 32 кГц и глубиной 16 бит, моно, используется sox
    command_line = "echo '" + text + "' | RHVoice-client -s Anna+CLB " 
    command_line += "| sox -t wav - -r 32000 -c 1 -b 16 -t wav - >'" + os.path.dirname(os.path.realpath(__file__)) + '/' + filename_audio + "'"
    if mode == 'playback':
        subprocess.call(command_line, shell=True)
        sound = AudioSegment.from_wav(os.path.dirname(os.path.realpath(__file__)) + '/' + filename_audio)
        play(sound)
        # command_line += "| aplay" - не используется, т.к. при каждом обращении выводится сообщение от самого RHVoice в терминал
    elif mode == 'into_file':
        subprocess.call(command_line, shell=True)
    else:
        print('[E] Неподдерживаемый режим работы, проверьте значение переменной mode.')
        return
     
    # Если вылазит FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe', нужно выполнить apt-get install ffmpeg libavcodec-extra




def main():
    tts('ты любишь меня?', 'into_file', 'temp/answer.wav')


if __name__ == '__main__':
    main()
        

''' 
В данном варианте нужно как-то ждать завершения воспроизведения, т.к. выполнения основного потока не прерывается.
Тоже самое при использовании pygame.mixer
import pyglet
explosion = pyglet.media.load('/home/vladislav/Проекты/test.wav', streaming=False)
explosion.play()
pyglet.app.run() '''