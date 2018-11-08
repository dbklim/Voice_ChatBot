# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для распознавания речи с помощью PocketSphinx.
'''

import os
import sys
import re
import subprocess
from pydub import AudioSegment
from pocketsphinx import Pocketsphinx, LiveSpeech, get_model_path


class SpeechRecognition:
    ''' Предназначен для распознавания речи с помощью PocketSphinx.
    1. mode - может иметь два значения: from_file и from_microphone
    1.1. from_file - распознавание речи из .wav файла (частота дискретизации 16кГц, 16bit, моно)
    1.2. from_microphone - распознавание речи с микрофона '''
    def __init__(self, mode):
        self.current_dirname = os.path.dirname(os.path.realpath(__file__))
        self.work_mode = mode
        model_path = get_model_path()

        if self.work_mode == 'from_file':
            config = {
                'hmm': os.path.join(model_path, 'zero_ru.cd_cont_4000'),
                'lm': os.path.join(model_path, 'ru_bot.lm'),
                'dict': os.path.join(model_path, 'ru_bot.dic')
            }
            self.speech_from_file = Pocketsphinx(**config)
        elif self.work_mode == 'from_microphone':
            self.speech_from_microphone = LiveSpeech(
                verbose=False,
                sampling_rate=16000,
                buffer_size=2048,
                no_search=False,
                full_utt=False,
                hmm=os.path.join(model_path, 'zero_ru.cd_cont_4000'),
                lm=os.path.join(model_path, 'ru_bot.lm'),
                dic=os.path.join(model_path, 'ru_bot.dic')
            )
        else:
            print('[E] Неподдерживаемый режим работы, проверьте значение переменной mode.')
            return

    # Добавить фильтры шума, например с помощью sox

    def stt(self, filename_audio = None):
        ''' Распознавание речи с помощью PocketSphinx. Режим задаётся при создании объекта класса (из файла или с микрофона).
        1. filename_audio - имя .wav или .opus файла с речью (частота дискретизации >=16кГц, 16bit, моно)
        2. возвращает строку с распознанной речью '''

        if self.work_mode == 'from_file':            
            if filename_audio == None:
                print('[E] В режиме from_file необходимо указывать имя .wav или .opus файла.')
                return
            filename_audio_raw = filename_audio[:filename_audio.find('.')] + '.raw'
            filename_audio_wav = filename_audio[:filename_audio.find('.')] + '.wav'
            audio_format = filename_audio[filename_audio.find('.') + 1:]
            
            # Конвертирование .opus файла в .wav
            if audio_format == 'opus': 
                command_line = "yes | ffmpeg -i '" + filename_audio + "' '" + filename_audio_wav + "'"
                proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = proc.communicate()
                if err.decode().find(filename_audio + ':') != -1:
                    return 'error'

            # Конвертирование .wav файла в .raw
            audio_file = AudioSegment.from_wav(self.current_dirname + '/' + filename_audio_wav)
            audio_file = audio_file.set_frame_rate(16000)
            audio_file.export(self.current_dirname + '/' + filename_audio_raw, format = 'raw')

            # Создание декодера и распознавание
            self.speech_from_file.decode(
                audio_file = self.current_dirname + '/' + filename_audio_raw,
                buffer_size = 2048,
                no_search = False,
                full_utt = False
            )
            return self.speech_from_file.hypothesis()
        elif self.work_mode == 'from_microphone':
            for phrase in self.speech_from_microphone:
                return str(phrase)


def main():   
    print('[i] Загрузка языковой модели... ', end = '')
    sr = SpeechRecognition('from_microphone')
    print('ок')
    while True:
        print(sr.stt())


if __name__ == '__main__':
    main()


'''
# Распознавание из потока данных с помощью pocketsphinx
from pocketsphinx import DefaultConfig, Decoder
    
model_path = get_model_path()

config = DefaultConfig()
config.set_string('-hmm', os.path.join(model_path, 'zero_ru.cd_cont_4000'))
config.set_string('-lm', os.path.join(model_path, 'ru.lm'))
config.set_string('-dict', os.path.join(model_path, 'my_dict.dic'))
    
decoder = Decoder(config)

# Decode streaming data
buf = bytearray(1024)
with open(os.path.join(os.path.dirname(sys.argv[0]) + '/data/answer.raw'), 'rb') as f:
    decoder.start_utt()
    while f.readinto(buf):
        decoder.process_raw(buf, False, False)
    decoder.end_utt()
segs = decoder.seg()
print('Best hypothesis segments:', [seg.word for seg in segs])
'''

'''
# Распознавание с помощью google speech cloud api
import speech_recognition as sr
    
r = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Скажите что-нибудь")
        audio = r.listen(source)
        print('Распознаю')
    
    try:
        print(r.recognize_google(audio, language="ru-RU"))
    except sr.UnknownValueError:
        print("Робот не расслышал фразу")
    except sr.RequestError as e:
        print("Ошибка сервиса; {0}".format(e))
'''
    
'''
# Распознавание с помощью wit.ai
from wit import Wit
import speech_recognition as sr   

client = Wit('4EXNGIL4JFS5NPKRZIRQWXAOU5DCKZRS')
   
r = sr.Recognizer()

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    print("Скажите что-нибудь")
    audio = r.listen(source)
    print('Распознаю')

resp = None
resp = client.speech(audio.get_wav_data(), None, {'Content-Type': 'audio/wav'})
print('Yay, got Wit.ai response: ' + str(resp))'''