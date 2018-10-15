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

'''
Потом дописать bot.py, что бы можно было аргументами из .sh запустить или обучение, или работу бота. Потом отредактировать диалоги пьесы, обучить
заново и протестировать. Затем в bot.py добавить распознавание голоса с микрофона и ответ голосом.
'''


f_source_data = 'data/source_data.txt'
f_prepared_data = 'data/prepared_data.txt'
f_prepared_data_pkl = 'data/prepared_data.pkl'
f_encoded_data = 'data/encoded_data.npz'
f_w2v_model = 'data/w2v_model.bin'
f_w2v_vocab = 'data/w2v_vocabulary.txt'
f_w2v_nbhd = 'data/w2v_neighborhood.txt'
f_net_model = 'data/net_model.txt'
f_net_weights = 'data/net_50_weights.h5'

def train():
    prep = Preparation()
    prep.prepare_all(f_source_data, f_prepared_data)

    w2v = CoderW2V()
    w2v.words2vec(f_prepared_data_pkl, f_encoded_data, f_w2v_model, f_w2v_vocab, f_w2v_nbhd, size = 150, epochs = 100) # 500 1000 для авторских данных

    t = Training()
    t.train(f_encoded_data, f_net_model, 2, 50, 5)

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