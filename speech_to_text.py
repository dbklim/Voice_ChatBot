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
import subprocess
from pydub import AudioSegment
from pocketsphinx import Pocketsphinx, LiveSpeech, get_model_path


class SpeechRecognition:
    ''' Предназначен для распознавания речи с помощью PocketSphinx.
    1. mode - может иметь два значения: from_file и from_microphone
    1.1. from_file - распознавание речи из .wav файла (частота дискретизации 16кГц, 16bit, моно)
    1.2. from_microphone - распознавание речи с микрофона '''
    def __init__(self, mode):
        self.current_dirname = os.path.dirname(sys.argv[0])
        self.work_mode = mode
        model_path = get_model_path()

        if self.work_mode == 'from_file':
            config = {
                'hmm': os.path.join(model_path, 'zero_ru.cd_cont_4000'),
                'lm': os.path.join(model_path, 'ru.lm'),
                'dict': os.path.join(model_path, 'ru_bot.dic')
            }
            self.speech_from_file = Pocketsphinx(**config)
        elif self.work_mode == 'from_microphone':
            self.speech_from_microphone = LiveSpeech(
                verbose=False,
                sampling_rate=8000,
                buffer_size=2048,
                no_search=False,
                full_utt=False,
                hmm=os.path.join(model_path, 'zero_ru.cd_cont_4000'),
                lm=os.path.join(model_path, 'ru.lm'),
                dic=os.path.join(model_path, 'ru_bot.dic')
            )
        else:
            print('[E] Неподдерживаемый режим работы, проверьте значение переменной mode.')
            return

    def stt(self, filename_audio = None):
        ''' Распознавание речи с помощью PocketSphinx. Режим задаётся при создании объекта класса (из файла или с микрофона).
        1. filename_audio - имя .wav файла с речью (частота дискретизации 16кГц, 16bit, моно)
        2. возвращает строку с распознанной речью '''

        if self.work_mode == 'from_file':            
            if filename_audio == None:
                print('[E] В режиме from_file необходимо указывать имя .wav файла.')
                return
            filename_audio_raw = filename_audio[:filename_audio.find('.')] + '.raw'
            # Конвертирование .wav файла в .raw
            audio_file = AudioSegment.from_wav(self.current_dirname + '/' + filename_audio) 
            audio_file.export(self.current_dirname + '/' + filename_audio_raw, format = 'raw')

            # Создание декодера и распознавание
            self.speech_from_file.decode(
                audio_file = self.current_dirname + '/' + filename_audio_raw,
                buffer_size = 1024,
                no_search = False,
                full_utt = False
            )
            return self.speech_from_file.hypothesis()
        elif self.work_mode == 'from_microphone':
            for phrase in self.speech_from_microphone:
                return phrase
        else:
            print('[E] Перед использованием данной функции необходимо вызвать initialize().')
            return


def create_dictionary(filename_vocabulary):
    ''' Создание словаря для pocketsphinx на основе словаря word2vec. После выполнения в .../pocketsphinx/model/ будет создан ru_bot.dic.
    1. filename_vocabulary - имя словаря word2vec '''
    filename_vocabulary = os.path.dirname(sys.argv[0]) + '/' + filename_vocabulary
    path_dict_for_ps = '/usr/local/lib/python3.5/dist-packages/pocketsphinx/model/'
    filename_dict_for_ps = 'ru_bot.dic'
    command_line = "sudo perl text2dict/dict2transcript.pl '" + filename_vocabulary + "' " + path_dict_for_ps + filename_dict_for_ps
    subprocess.call(command_line, shell=True)
    if os.path.exists(path_dict_for_ps + filename_dict_for_ps + '.accent'):
        subprocess.call('sudo rm -f ' + path_dict_for_ps + filename_dict_for_ps + '.accent', shell=True)




def main():
    print('[i] Загрузка языковой модели...')
    sr = SpeechRecognition('from_file')
    print('[i] Готово')
    print(sr.stt('data/answer.wav'))


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