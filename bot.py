# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

from preprocessing import Preparation
from coder_w2v import CoderW2V
from training import Training
from prediction import Prediction

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
    w2v.words2vec(f_prepared_data_pkl, f_encoded_data, f_w2v_model, f_w2v_vocab, f_w2v_nbhd, size = 500, epochs = 1000)

    t = Training()
    t.train(f_encoded_data, f_net_model, 2, 50, 10)   

    pr = Prediction(f_net_model, f_net_weights, f_w2v_model)
    pr.assessment_training_accuracy(f_encoded_data)

    print('[i] Общее время обучения: %.4f мин или %.4f ч' % ((time.time() - start_time)/60.0, ((time.time() - start_time)/60.0)/60.0))
    
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

def predict():
    pr = Prediction(f_net_model, f_net_weights, f_w2v_model)
    print('\n')
    while(True):
        quest = input("Пользователь: ")
        answer = pr.predict(quest)
        print("\t=> %s\n" % answer)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            predict()
        else:
            print('[E] Неверный аргумент командной строки. Поддерживаемые варианты:')
            print('\ttrain - обучение модели seq2seq')
            print('\tpredict - работа с обученной моделью seq2seq')
            sys.exit(0)
    else:
        curses.setupterm()
        print('[i] Выберите вариант работы бота:')
        print('\t1. train - обучение модели seq2seq')
        print('\t2. predict - работа с обученной моделью seq2seq')
        while True:
            choice = input('Введите цифру: ')
            if choice == '1':
                train()
                break
            elif choice == '2':
                predict()
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