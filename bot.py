#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Вопросно-ответный бот на основе нейросетевой модели seq2seq. Поддерживает общение
в текстовом формате, с синтезом (RHVoice) и распознаванием (PocketSphinx) речи.
'''

import sys
import signal
import platform
import curses
import os
from text_to_text import TextToText
from text_to_speech import TextToSpeech
from speech_to_text import SpeechToText
from preparing_speech_to_text import LanguageModel


f_name_source_data = 'data/plays_ru/plays_ru.txt'
f_name_training_sample = 'data/plays_ru/prepared_plays_ru.pkl'
f_name_enc_training_sample = 'data/plays_ru/encoded_plays_ru.npz'
f_name_w2v_model = 'data/plays_ru/w2v_model_plays_ru.bin'
f_name_w2v_vocab = 'data/plays_ru/w2v_vocabulary_plays_ru.txt'
f_name_model = 'data/plays_ru/model_plays_ru.json'
f_name_model_weights = 'data/plays_ru/model_weights_plays_ru.h5'


curses.setupterm()


def configure_file_names():
    ''' Запрашивает выбор одного из наборов данных (plays_ru, conversations_ru и subtitles_ru), корректирует имена файлов
    и возвращает имя выбранного набора данных. '''
    print('[i] Выберите набор данных:')
    print('\t1. plays_ru - набор диалогов из пьес')
    print('\t2. conversations_ru - набор диалогов из различных произведений')
    print('\t3. subtitles_ru - набор диалогов из субтитров к 347 сериалам')
    print('[W] conversations_ru и subtitles_ru ещё в разработке!')
    while True:
        choice = input('\rВаш выбор: ')
        if choice == '1':
            name_dataset = 'plays_ru'
            break
        elif choice == '2':
            name_dataset = 'conversations_ru'
            break
        elif choice == '3':
            name_dataset = 'subtitles_ru'
            break
        else:
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
            print('                                     ', end='')

    global f_name_source_data, f_name_training_sample, f_name_enc_training_sample, f_name_w2v_model, f_name_w2v_vocab, f_name_model, f_name_model_weights

    f_name_source_data = 'data/' + name_dataset + '/' + name_dataset + '.txt'
    f_name_training_sample = 'data/' + name_dataset + '/prepared_' + name_dataset + '.pkl'
    f_name_enc_training_sample = 'data/' + name_dataset + '/encoded_' + name_dataset + '.npz'
    f_name_w2v_model = 'data/' + name_dataset + '/w2v_model_' + name_dataset + '.bin'
    f_name_w2v_vocab = 'data/' + name_dataset + '/w2v_vocabulary_' + name_dataset + '.txt'
    f_name_model = 'data/' + name_dataset + '/model_' + name_dataset + '.json'
    f_name_model_weights = 'data/' + name_dataset + '/model_weights_' + name_dataset + '.h5'
    return name_dataset


def train():
    ''' Обучение модели seq2seq. '''
    name_dataset = configure_file_names()

    f_name_subtitles = None
    f_name_prepared_subtitles = None
    if name_dataset != 'subtitles_ru':
        f_name_subtitles = 'data/subtitles_ru/subtitles_ru.txt'
        f_name_prepared_subtitles = 'data/subtitles_ru/prepared_subtitles_ru.pkl'

    ttt = TextToText(train=True)
    ttt.prepare(f_name_source_data, f_name_training_sample, f_name_subtitles, f_name_prepared_subtitles, f_name_enc_training_sample, f_name_w2v_model, 
                f_name_w2v_vocab, len_encode=5000, size=500, epochs=500, logging=True)
    ttt.train(f_name_enc_training_sample, f_name_model, f_name_model_weights, training_cycles=200, epochs=5)

    lm = LanguageModel()
    lm.build_language_model(f_name_source_data, 5000)
    
    # SimpleSeq2Seq
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

    # AttentionSeq2Seq
    # 100 итераций, 5 эпох - точность 99.31% (1590 из 1601 правильных ответов), ошибка 0.0120 (размер вектора 500)


def predict(speech_recognition=False, speech_synthesis=False):
    ''' Работа с обученной моделью seq2seq.
    1. speech_recognition - включение распознавания речи с микрофона с помощью PocketSphinx
    2. speech_synthesis - включение озвучивания ответов с помощью RHVoice '''
    name_dataset = configure_file_names()

    ttt = TextToText(f_name_w2v_model=f_name_w2v_model, f_name_model=f_name_model, f_name_model_weights=f_name_model_weights)

    if speech_recognition:
        print('[i] Загрузка языковой модели для распознавания речи...')
        stt = SpeechToText('from_microphone', name_dataset)

    if speech_synthesis:
        print('[i] Загрузка синтезатора речи...')
        tts = TextToSpeech('anna')

    print()
    question = ''
    while(True):
        if speech_recognition:
            print('Слушаю...')
            question = stt.get()
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
            print('Вы: ' + question)
        else:
            question = input('Вы: ')
        answer, lost_words = ttt.predict(question, True)
        print('\t=> %s' % answer)
        if len(lost_words) > 0:
            print('[w] Потерянные слова: ' + ', '.join(lost_words) + '\n')
        else:
            print()
        if speech_synthesis:
            tts.get(answer)

# Попробовать в качестве центральной части использовать pynlc или chatterbot (что бы была поддержка сценария и некоторого контекста, который можно
# брать из БД)

def main():    
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
            print('По умолчанию используется набор диалогов из пьес plays_ru.')
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
            choice = input('\rВаш выбор: ')
            if choice == '1':
                choice = input('Вы уверены?(д/н) ')
                if choice == 'д' or choice == 'y':
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
                print('                                     ', end='')




def on_stop(*args):
    print('\n[i] Бот остановлен')
    sys.exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == 'Linux':
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    main()