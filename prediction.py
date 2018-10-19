# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для работы с обученной нейронной сетью seq2seq.
'''

from preprocessing import Preparation
from coder_w2v import CoderW2V

import seq2seq
from seq2seq.models import SimpleSeq2Seq
import numpy as np
import re
import sys
import curses
import os


class Prediction:
    ''' Предназначен для работы с обученной нейронной сетью seq2seq. 
    1. filename_net_model - имя .txt файла с параметрами модели SimpleSeq2Seq
    2. filename_net_weights - имя .h5 файла с весами обученной сети
    3. filename_w2v_model - имя .bin файла с обученной W2V моделью '''
    def __init__(self, filename_net_model, filename_net_weights, filename_w2v_model):
        print('[i] Загрузка параметров сети из %s и %s' % (filename_net_model, filename_net_weights))

        _input_dim, _hidden_dim, _output_length, _output_dim, _depth, _loss, _optimizer = self.__load_simpleseq2seq_model(filename_net_model)
        
        self.model = SimpleSeq2Seq(input_dim = _input_dim, hidden_dim = _hidden_dim, output_length = _output_length, output_dim = _output_dim, depth = _depth)
        self.model.compile(loss = _loss, optimizer = _optimizer)
        self.model.load_weights(filename_net_weights)

        self.preparation = Preparation()
        self.preparation.input_size = _output_length
        self.w2v = CoderW2V('load_model', filename_w2v_model)

    def predict(self, quest):
        ''' Предварительная обработка вопроса к сети, перевод его в вектор, получение ответа от сети и перевод его в строку.
        1. возвращает строку с ответом сети '''
        quest = self.preparation.prepare_quest(quest)
        quest = self.w2v.word2vec(quest)            
        if isinstance(quest, str) == True:
            return "я не знаю слова '%s', извините" % quest[quest.find(':') + 1:]
        quest = (quest + 1.0) * 0.5
        answer = self.model.predict(quest[np.newaxis,:])        
        answer = answer * 2.0 - 1.0  
        answer = self.w2v.vec2word(answer[0])
        answer = self.preparation.prepare_answer(answer)
        return answer

    def assessment_training_accuracy(self, filename):
        ''' Оценка точности обучения сети: подаёт на вход сети все вопросы из обучающей выборки и сравнивает полученный ответ сети с
        ответом из обучающей выборки. 
        1. filename - имя .npz файла с векторным представлением слов в парах [вопрос,ответ]'''
        print('[i] Оценка точности обучения сети...')

        npzfile = np.load(filename)
        Q, A = npzfile["Q"], npzfile["A"]
        
        Q = (Q + 1.0) * 0.5

        curses.setupterm()
        
        correct_answers = 0
        len_Q = len(Q)
        print('Оценено 0 из %i...' % len_Q)
        for i in range(len_Q):
            answer = self.model.predict(Q[i][np.newaxis,:])
            answer = answer * 2.0 - 1.0 
            answer = self.w2v.vec2word(answer[0])
            answer = self.preparation.prepare_answer(answer)
            answer_standart = self.w2v.vec2word(A[i])
            answer_standart = self.preparation.prepare_answer(answer_standart)
            if answer == answer_standart:
                correct_answers += 1     
            if i % 10 == 0:       
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('Оценено %i из %i, правильных ответов %i...' % (i, len_Q, correct_answers))  
        accuracy = (float(correct_answers)/float(len(A))) * 100
        print('[i] Количество правильных ответов %i из %i, итоговая точность %.2f%%' % (correct_answers, len(A), accuracy))

    def __load_simpleseq2seq_model(self, filename):
        ''' Загрузка параметров модели SimpleSeq2Seq и параметров компиляции (optimizer и loss) из .txt файла.  '''
        file_r = open(filename)
        parametrs = file_r.readlines()
        file_r.close()
        input_dim = hidden_dim = output_length = output_dim = depth = loss = optimizer = None
        for line in parametrs:
            if line.find('input_dim') != -1:
                input_dim = int(line[line.find('=') + 1:line.find('\n')])
            elif line.find('hidden_dim') != -1:
                hidden_dim = int(line[line.find('=') + 1:line.find('\n')])
            elif line.find('output_length') != -1:
                output_length = int(line[line.find('=') + 1:line.find('\n')])
            elif line.find('output_dim') != -1:
                output_dim = int(line[line.find('=') + 1:line.find('\n')])
            elif line.find('depth') != -1:
                depth = int(line[line.find('=') + 1:line.find('\n')])
            elif line.find('loss') != -1:
                loss = line[line.find('=') + 1:line.find('\n')]
            elif line.find('optimizer') != -1:
                optimizer = line[line.find('=') + 1:line.find('\n')]
        return input_dim, hidden_dim, output_length, output_dim, depth, loss, optimizer




def main():
    pr = Prediction('data/net_model.txt', 'data/net_50_weights.h5', 'data/w2v_model.bin')
    while(True):
        quest = input("Пользователь: ")
        answer = pr.predict(quest)
        print("\t=> %s\n" % answer)


if __name__ == '__main__':
    main()