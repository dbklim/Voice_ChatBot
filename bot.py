#!/usr/bin/python3 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

from preprocessing import Preparation
from coder_w2v import CoderW2V
from training import Training
from prediction import Prediction
from text_to_speech import tts
from speech_to_text import SpeechRecognition
from preparing_speech_to_text import building_language_model

import sys
import signal
import platform
import curses
import os
import time


f_source_data = 'data/source_data.txt'
f_prepared_data = 'data/prepared_data.txt'
f_prepared_data_pkl = 'data/prepared_data.pkl'
f_encoded_data = 'data/encoded_data.npz'
f_w2v_model = 'data/w2v_model.bin'
f_w2v_vocab = 'data/w2v_vocabulary.txt'
f_w2v_nbhd = 'data/w2v_neighborhood.txt'
f_net_model = 'data/net_model.txt'
f_net_weights = 'data/net_final_weights.h5'


def train():
    start_time = time.time()
    prep = Preparation()
    prep.prepare_all(f_source_data, f_prepared_data)

    w2v = CoderW2V()
    w2v.words2vec(f_prepared_data_pkl, f_encoded_data, f_w2v_model, f_w2v_vocab, f_w2v_nbhd, size = 500, epochs = 1000, window=5)

    t = Training()
    t.train(f_encoded_data, f_net_model, 2, 150, 5)       

    pr = Prediction(f_net_model, f_net_weights, f_w2v_model)
    pr.assessment_training_accuracy(f_encoded_data)

    print('\n[i] Общее время обучения: %.4f мин или %.4f ч' % ((time.time() - start_time)/60.0, ((time.time() - start_time)/60.0)/60.0))
    
    building_language_model(f_source_data)
    
    # 248 входных фраз, epochs = 100
    # 10 итераций, 5 эпох - точность 0% (size = 150)
    # 50 итераций, 5 эпох - точность 22.58% (size = 150)
    # 50 итераций, 10 эпох - точность 61.29% (size = 150)
    # 50 итераций, 10 эпох - точность 69.35% (size = 200)

    # 50 итераций, 15 эпох - точность 64.52% (size = 100)
    # 50 итераций, 15 эпох - точность 83.87% (size = 150)
    # 50 итераций, 15 эпох - точность 80.65% (size = 200)

    # 50 итераций, 20 эпох - точность 95.16% (size = 150)
    # 50 итераций, 20 эпох - точность 98.39% (size = 200) (244 из 248 правильных ответов)

    # 50 итераций, 25 эпох - точность 96.77% (size = 150)
    # 50 итераций, 25 эпох - точность 99% (size = 200) (246 из 248 правильных ответов)

    # 50 итераций, 30 эпох - точность 96.77% (size = 150) (240 из 248 правильных ответов)
    # 50 итераций, 30 эпох - точность 100% (size = 200) (248 из 248 правильных ответов)

    # 1596 входных фраз, epochs = 1000, 50 итераций по 30 эпох
    # size = 1250 - недостаточно видеопамяти (ошибка аллокации)
    # size = 1100 - недостаточно видеопамяти
    # size = 1000 - недостаточно видеопамяти
    # size = 900 - обучение займёт примерно 44.5 часов
    # size = 800 - обучение займёт примерно 33 часа
    # size = 700 - обучение займёт примерно 25.8 часа
    # size = 600 - обучение займёт примерно 21.2 часа
    # size = 500 - обучение займёт примерно 15.4 часа

    # 1596 входных фраз, epochs = 1000, size = 500
    # 50 итераций, 20 эпох - обучение займёт примерно 9.4 часа
    # 50 итераций, 15 эпох - обучение займёт примерно 7.3 часа
    # 50 итераций, 10 эпох - точность 98.62% (1574 из 1596 правильных ответов), время обучения - 4.8ч или 287.7 мин
    # 50 итераций, 10 эпох - точность 98.18% (1567 из 1596 правильных ответов)
    # 55 итераций, 10 эпох - точность 98.38% (1575 из 1601 правильных ответов)
    # 55 итераций, 5 эпох - точность 94.57% (1514 из 1601 првильных ответов)
    # 55 итераций, 5 эпох - точность 95.19% (1524 из 1601 правильных ответов)
    # 150 итераций, 5 эпох - точность 99.25% (1589 из 1601 правильных ответов), ошибка 0.207


    # window = 10, 65 итераций, 5 эпох - точность 0.87% (14 из 1601 правильных ответов), ошибка 0.0957
    # window = 10, 150 итераций, 5 эпох - точность 98.5% (1577 из 1601 правильных ответов), ошибка 0.0348

def predict(for_speech_recognition = False, for_speech_synthesis = False):
    ''' Работа с обученной моделью seq2seq.
    1. for_speech_recognition - включение распознавания речи с микрофона с помощью PocketSphinx
    2. for_speech_synthesis - включение озвучивания ответов с помощью RHVoice '''

    pr = Prediction(f_net_model, f_net_weights, f_w2v_model)

    if for_speech_recognition:
        print('[i] Инициализация языковой модели...')
        sr = SpeechRecognition('from_microphone')

    print('\n')
    quest = ''
    while(True):
        if for_speech_recognition == True:        
            print('Слушаю...')
            quest = sr.stt()
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
            print('Пользователь: ' + quest)
        else:
            quest = input("Пользователь: ")
        answer = pr.predict(quest)
        print("\t=> %s\n" % answer)
        if for_speech_synthesis == True:
            tts(answer, 'playback')

# Попробовать в качестве центральной сети использовать pynlc или chatterbot (что бы была поддержка сценария и некоторого контекста, который можно 
# брать из БД)

def main():
    curses.setupterm()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            if len(sys.argv) > 2:
                if sys.argv[2] == '-ss':
                    if len(sys.argv) > 3:
                        if sys.argv[3] == '-sr':
                            predict(True, True)
                        else:
                            print("[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.")
                            sys.exit(0)
                    predict(False, True)
                    
                elif sys.argv[2] == '-sr':
                    if len(sys.argv) > 3:
                        if sys.argv[3] == '-ss':
                            predict(True, True) 
                        else:
                            print("[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.")
                            sys.exit(0)
                    predict(True, False)

                else:
                    print("[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.")
                    sys.exit(0)                    
            predict()
        elif sys.argv[1] == 'help':
            print('Поддерживаемые варианты работы:')
            print('\ttrain - обучение модели seq2seq')
            print('\tpredict - работа с обученной моделью seq2seq')
            print('\tpredict -ss - включено озвучивание ответов с помощью RHVoice')
            print('\tpredict -sr - включено распознавание речи с помощью PocketSphinx')
            print('\tpredict -ss -sr - включено озвучивание ответов и распознавание речи')
            print('\tpredict -sr -ss - включено озвучивание ответов и распознавание речи')
            sys.exit(0)
        else:
            print("[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.")
            sys.exit(0)
    else:
        print('[i] Выберите вариант работы бота:')
        print('\t1. train - обучение модели seq2seq')
        print('\t2. predict - работа с обученной моделью seq2seq')
        print('\t3. predict -ss - включено озвучивание ответов с помощью RHVoice')
        print('\t4. predict -sr - включено распознавание речи с помощью PocketSphinx')
        print('\t5. predict -ss -sr - включено озвучивание ответов и распознавание речи')
        while True:
            choice = input('Введите цифру: ')
            if choice == '1':
                choice = input('Вы уверены?(д/н) ')
                if choice == 'д':
                    train()
                break
            elif choice == '2':
                predict()
                break
            elif choice == '3':
                predict(False, True)
                break
            elif choice == '4':
                predict(True, False)
                break
            elif choice == '5':
                predict(True, True)
                break
            else:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))




def on_stop(*args):
    print("\n[i] Бот остановлен")
    sys.exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == "Linux":
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    main()