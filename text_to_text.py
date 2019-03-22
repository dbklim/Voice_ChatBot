#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Обучение и взаимодействие с нейросетевой моделью seq2seq для генерации ответа на вопрос.
Используется модель AttentionSeq2Seq из библиотеки seq2seq.
'''

import os
import platform
import signal
import sys
import time
import curses
import json
import numpy as np
from keras import __version__ as keras_version
from keras.models import model_from_json
from seq2seq.models import AttentionSeq2Seq
from seq2seq.cells import LSTMDecoderCell, AttentionDecoderCell
from recurrentshop import RecurrentSequential
from recurrentshop.engine import _OptionalInputPlaceHolder

from source_to_prepared import SourceToPrepared
from word_to_vec import WordToVec


curses.setupterm()


class TextToText:
    ''' Позволяет обучать модель AttentionSeq2Seq для генерации ответа на вопрос и взаимодействовать с этой обученной моделью.
    1. name_dataset - имя используемого набора данных: plays_ru, subtitles_ru или conversations_ru
    2. f_name_w2v_model - имя .bin файла с обученной моделью wor2vec (по умолчанию data/+name_dataset+/w2v_model_+name_dataset+.bin)
    3. f_name_model - имя .json файла с моделью сети (по умолчанию data/+name_dataset+/model_+name_dataset+.json)
    4. f_name_model_weights - имя .h5 файла с весами обученной модели (по умолчанию data/+name_dataset+/model_weights_+name_dataset+.h5)
    5. train - True: обучение модели с нуля, False: взаимодействие с обученной моделью '''
    def __init__(self, name_dataset='plays_ru', f_name_w2v_model=None, f_name_model=None, f_name_model_weights=None, train=False):
        self.stp = None
        self.w2v = None
        self.model = None
        if not train:
            if not (name_dataset == 'plays_ru' or name_dataset == 'subtitles_ru' or name_dataset == 'conversations_ru'):
                print('\n[E] Неверное значение name_dataset. Возможные варианты: plays_ru, subtitles_ru или conversations_ru\n')
                return

            print('[i] Используется набор данных ' + name_dataset)
            if f_name_w2v_model is None:
                f_name_w2v_model = 'data/' + name_dataset + '/w2v_model_' + name_dataset + '.bin'
                if not os.path.isfile(f_name_w2v_model):
                    print("\n[E] Файл '" + f_name_w2v_model + "' не существует\n")
                    return
            if f_name_model is None:
                f_name_model = 'data/' + name_dataset + '/model_' + name_dataset + '.json'
                if not os.path.isfile(f_name_model):
                    print("\n[E] Файл '" + f_name_model + "' не существует\n")
                    return
            if f_name_model_weights is None:
                f_name_model_weights = 'data/' + name_dataset + '/model_weights_' + name_dataset + '.h5'
                if not os.path.isfile(f_name_model_weights):
                    print("\n[E] Файл '" + f_name_model_weights + "' не существует\n")
                    return
            
            print('[i] Загрузка параметров модели из %s и %s' % (f_name_model, f_name_model_weights))
            self.__load_model(f_name_model)
            self.model.load_weights(f_name_model_weights)
            
            self.stp = SourceToPrepared(self.model.get_layer(index=0).input_shape[1])
            self.w2v = WordToVec(f_name_w2v_model)


    def prepare(self, f_name_source_data, f_name_training_sample=None, f_name_source_subtitles=None, f_name_prepared_subtitles=None,
                f_name_enc_training_sample=None, f_name_w2v_model=None, f_name_w2v_vocab=None, len_encode=None, size=500, min_count_repeat=1,
                window=5, epochs=None, logging=False):
        ''' Осуществляет преобразование пар "вопрос %% ответ" в последовательности фиксированного размера. Затем выполняет построение модели word2vec
        и кодирования обучающей выборки в вектора.
        1. f_name_source_data - имя входного .txt файла с исходными данными (пары "вопрос %% ответ")
        2. f_name_training_sample - имя .pkl файла с предварительно обработанными парами [вопрос,ответ] (по умолчанию prepared_+f_name_source_data+.pkl)
        3. f_name_source_subtitles - имя входного .txt файла с субтитрами (пары "вопрос %% ответ")
        4. f_name_prepared_subtitles - имя .pkl файла с предварительно обработанными субтитрами (для расширения словаря модели word2vec) (по умолчанию prepared_+f_name_source_subtitles+.pkl)
        5. f_name_enc_training_sample - имя выходного .npz файла с векторным представлением слов в парах [вопрос,ответ] (по умолчанию encoded_+f_name_training_sample+.npz)
        6. f_name_w2v_model - имя .bin файла для сохранения обученной модели word2vec (по умолчанию w2v_model_+f_name_training_sample+.bin)
        7. f_name_w2v_vocab - имя .txt файла для сохранения словаря word2vec (по умолчанию w2v_vocabulary_+f_name_training_sample+.txt)
        8. len_encode - если None: закодировать весь f_name_training_sample, иначе - первые len_encode элементов
        9. size - рразмерность вектора, которым кодируется одно слово
        Обычно используют значения в несколько сотен, большие значения требуют больше данных обучения, но могут привести к более точным моделям.
        10. min_count - минимальное количество повторений (частота повторений) слова, что бы оно было включено в словарь
        11. window - максимальное расстояние между текущим и прогнозируемым словом в предложении
        12. epochs - число эпох обучения модели word2vec (если None, используется внутренний итератор)
        13. logging - включение вывода данных в процессе обучения модели word2vec '''
        
        if not os.path.isfile(f_name_source_data):
            print("\n[E] Файл '" + f_name_source_data + "' не существует\n")
            return
        
        start_time = time.time()
        self.stp = SourceToPrepared()
        if f_name_training_sample is None:
            f_name_training_sample = f_name_source_data[:f_name_source_data.rfind('/')+1] + 'prepared_' + f_name_source_data[f_name_source_data.rfind('/')+1:]
            f_name_training_sample = f_name_training_sample.replace('.txt', '.pkl')
        self.stp.prepare_all(f_name_source_data, f_name_training_sample)

        if f_name_source_subtitles is not None:
            stp_subtitles = SourceToPrepared()
            if f_name_prepared_subtitles is None:
                f_name_prepared_subtitles = f_name_source_subtitles[:f_name_source_subtitles.rfind('/')+1] + 'prepared_' + \
                                            f_name_source_subtitles[f_name_source_subtitles.rfind('/')+1:]
                f_name_prepared_subtitles = f_name_prepared_subtitles.replace('.txt', '.pkl')
            stp_subtitles.prepare_all(f_name_source_subtitles, f_name_prepared_subtitles)

        self.w2v = WordToVec()
        self.w2v.build_word2vec(f_name_training_sample, f_name_prepared_subtitles, f_name_enc_training_sample, f_name_w2v_model, f_name_w2v_vocab, len_encode,
                                size, min_count_repeat, window, epochs, logging)
        print('[i] Время обработки: %.2f мин или %.2f ч' % ((time.time() - start_time)/60.0, ((time.time() - start_time)/60.0)/60.0))


    def load_prepared(self, name_dataset='plays_ru', f_name_w2v_model=None, f_name_enc_training_sample=None):
        ''' Установка максимальной длины предложения (размера входа) для модели сети и загрузка обученной модели word2vec.
        1. name_dataset - имя используемого набора данных: plays_ru, subtitles_ru или conversations_ru
        2. f_name_w2v_model - имя .bin файла с обученной моделью wor2vec (по умолчанию data/+name_dataset+/w2v_model_+name_dataset+.bin)
        3. f_name_enc_training_sample - имя .npz файла с векторным представлением слов в парах [вопрос,ответ] (по умолчанию data/+name_dataset+/encoded_+name_dataset+.npz) '''

        if not (name_dataset == 'plays_ru' or name_dataset == 'subtitles_ru' or name_dataset == 'conversations_ru'):
            print('\n[E] Неверное значение name_dataset. Возможные варианты: plays_ru, subtitles_ru или conversations_ru\n')
            return

        print('[i] Используется набор данных ' + name_dataset)
        if f_name_w2v_model is None:
            f_name_w2v_model = 'data/' + name_dataset + '/w2v_model_' + name_dataset + '.bin'
            if not os.path.isfile(f_name_w2v_model):
                print("\n[E] Файл '" + f_name_w2v_model + "' не существует\n")
                return
        if f_name_enc_training_sample is None:
            f_name_enc_training_sample = 'data/' + name_dataset + '/encoded_' + name_dataset + '.npz'
            if not os.path.isfile(f_name_enc_training_sample):
                print("\n[E] Файл '" + f_name_enc_training_sample + "' не существует\n")
                return

        print('[i] Загрузка данных из %s' % f_name_enc_training_sample)
        npzfile = np.load(f_name_enc_training_sample)
        questions = npzfile['questions']
        vec_size = questions.shape[1]
        self.stp = SourceToPrepared(vec_size)

        self.w2v = WordToVec(f_name_w2v_model)


    def train(self, f_name_enc_training_sample, f_name_model=None, f_name_model_weights=None, depth_model=2, training_cycles=100, epochs=5):
        ''' Запуск обучения и тестирования модели AttentionSeq2Seq.
        1. f_name_enc_training_sample - имя .npz файла с векторным представлением слов в парах [вопрос,ответ]
        2. f_name_model - имя .json файла для сохранения модели сети (по умолчанию model_+f_name_enc_training_sample+.json)
        3. f_name_model_weights - имя .h5 файла для сохранения весов обученной модели (по умолчанию model_weights_+f_name_enc_training_sample+.h5)
        4. depth_model - глубина модели seq2seq, задаёт число входных и выходных LSTM-слоёв
        5. training_cycles - количество циклов обучения модели
        6. epochs - количество эпох в одном цикле обучения модели '''

        if (self.stp is None and self.w2v is None) or not os.path.isfile(f_name_enc_training_sample):
            print('[E] Перед обучением модели сети необходимо подготовить обучающие данные с помощью prepare().')
            return

        if f_name_model is None:
            f_name_model = f_name_enc_training_sample.replace('encoded_', 'model_')
            f_name_model = f_name_model.replace('.npz', '.json')
        if f_name_model_weights is None:
            f_name_model_weights = f_name_enc_training_sample.replace('encoded_', 'model_weights_')
            f_name_model_weights = f_name_model_weights.replace('.npz', '.h5')

        start_time = time.time()
        print('[i] Загрузка данных из %s' % f_name_enc_training_sample)
        npzfile = np.load(f_name_enc_training_sample)
        questions, answers = npzfile['questions'], npzfile['answers']
        questions = (questions + 1.0) * 0.5
        answers = (answers + 1.0) * 0.5

        num_examples, sequence_length, vec_size = questions.shape
        print('\tколичество примеров: %i' % num_examples)
        print('\tдлинна последовательности: %i' % sequence_length)
        print('\tразмер входа: %i' % vec_size)

        print('[i] Построение сети...\n')
        #self.model = SimpleSeq2Seq(input_dim=vec_size, hidden_dim=vec_size, output_length=sequence_length, output_dim=vec_size, depth=depth_model)
        self.model = AttentionSeq2Seq(input_dim=vec_size, hidden_dim=vec_size, input_length=sequence_length, output_length=sequence_length,
                                      output_dim=vec_size, depth=depth_model, dropout=0.0)

        self.model.compile(loss='mse', optimizer='rmsprop')
        self.__save_model(f_name_model)
        print(self.model.summary())

        print('\n[i] Обучение сети...\n')
        for i in range(1, training_cycles + 1):
            self.model.fit(questions, answers, batch_size=32, epochs=epochs, verbose=1)
            self.__save_model_weights(f_name_model_weights, training_cycles, i)
        print('[i] Обучение завершено')
        
        print('[i] Оценка сети...')
        score = self.model.evaluate(questions, answers, batch_size=32, verbose=1)
        print('[i] Оценка точности модели на обучающей выборке: %.2f%%' % (score*100))

        self.assessment_training_accuracy(f_name_enc_training_sample)
        print('[i] Время обучения: %.2f мин или %.2f ч' % ((time.time() - start_time)/60.0, ((time.time() - start_time)/60.0)/60.0))
        

    def __save_compile_param(self, f_name_compile_param, loss, optimizer):
        ''' Сохранение параметров компиляции (loss и optimizer) модели сети в .json файл f_name_compile_param. '''
        with open(f_name_compile_param, 'w') as f_compile_param:
            json.dump({'loss':loss, 'optimizer':optimizer}, f_compile_param)

    
    def __load_compile_param(self, f_name_compile_param):
        ''' Загрузка параметров компиляции (loss и optimizer) модели сети из .json файла f_name_compile_param. '''
        with open(f_name_compile_param, 'r') as f_compile_param:
            param = json.load(f_compile_param)
        return param['loss'], param['optimizer']


    def __save_model(self, f_name_model):
        ''' Сохранение модели сети в .json файла f_name_model. '''
        model_json = self.model.to_json()
        with open(f_name_model, 'w') as f_model:
            f_model.write(model_json)
    

    def __load_model(self, f_name_model):
        ''' Загрузка модели сети из .json файла f_name_model. '''
        with open(f_name_model, 'r') as f_model:
            self.model = model_from_json(f_model.read(), custom_objects={'RecurrentSequential': RecurrentSequential, 
                                                                         '_OptionalInputPlaceHolder': _OptionalInputPlaceHolder,
                                                                         'LSTMDecoderCell': LSTMDecoderCell,
                                                                         'AttentionDecoderCell': AttentionDecoderCell})


    def __save_model_weights(self, f_name_model_weights, training_cycles, i):
        ''' Сохранение весов модели после каждой итерации обучения в .h5 файл f_name_model_weights. '''
        print('\n[i] Сохранение результата %i из %i...\n' % (i, training_cycles))
        if i == training_cycles:
            self.model.save_weights(f_name_model_weights)
            path_weights = os.path.join(os.path.abspath(os.path.dirname(__file__)), f_name_model_weights[:len(f_name_model_weights)-3] + '_%i.h5' % (i - 1))
            os.remove(path_weights)
        else:
            self.model.save_weights(f_name_model_weights[:len(f_name_model_weights)-3] + '_%i.h5' % i)
            if i > 1:
                path_weights = os.path.join(os.path.abspath(os.path.dirname(__file__)), f_name_model_weights[:len(f_name_model_weights)-3] + '_%i.h5' % (i - 1))
                os.remove(path_weights)

    
    def assessment_training_accuracy(self, f_name_enc_training_sample, f_name_wrong_answers=None):
        ''' Оценка точности обучения сети: подаёт на вход сети все вопросы из обучающей выборки и сравнивает полученный ответ сети с
        ответом из обучающей выборки. 
        1. f_name_enc_training_sample - имя .npz файла с векторным представлением слов в парах [вопрос,ответ]
        2. f_name_wrong_answers - имя .txt файла для сохранения неправильных ответов сети (по умолчанию data/wrong_answers.txt) '''

        if f_name_wrong_answers is None:
            f_name_wrong_answers = f_name_enc_training_sample.replace('encoded_', 'wrong_answers_')
            f_name_wrong_answers = f_name_wrong_answers.replace('.npz', '.txt')

        print('[i] Оценка точности обучения модели...')

        npzfile = np.load(f_name_enc_training_sample)
        questions, answers = npzfile['questions'], npzfile['answers']
        questions = (questions + 1.0) * 0.5
        
        correct_answers = 0
        wrong_answers = []
        len_questions = len(questions)
        print('[i] Оценено 0 из %i, правильных ответов 0, текущая точность 0.00%%' % len_questions)
        for i in range(len_questions):
            if i % 10 == 0 and i > 0:       
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Оценено %i из %i, правильных ответов %i, текущая точность %.2f%%' % (i, len_questions, correct_answers, correct_answers/i*100))
            answer = self.model.predict(questions[i][np.newaxis,:])
            answer = answer * 2.0 - 1.0
            answer = self.w2v.vec2word(answer[0])
            #answer = self.stp.prepare_answer(answer)
            answer_etalon = self.w2v.vec2word(answers[i])
            #answer_etalon = self.stp.prepare_answer(answer_etalon)
            if answer == answer_etalon:
            #if np.all(answer[0] == answers[i]):
                correct_answers += 1
            else:
                # Сохранение неправильных ответов для последующего вывода
                quest = self.w2v.vec2word(questions[i])
                quest = list(reversed(quest))
                quest = self.stp.prepare_answer(quest)
                
                #answer = self.w2v.vec2word(answer[0])
                answer = self.stp.prepare_answer(answer)
                
                wrong_answers.append([quest, answer])        
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Оценено %i из %i, правильных ответов %i, текущая точность %.2f%%' % (len_questions, len_questions, correct_answers, correct_answers/len_questions*100))

        accuracy = correct_answers / len(answers) * 100
        print('[i] Количество правильных ответов %i из %i, итоговая точность %.2f%%' % (correct_answers, len(answers), accuracy))

        if len(wrong_answers) < 50:
            i = 0
            print('[i] Неправильные ответы:')
            for phrase in wrong_answers:
                i += 1
                print('%i. %s  %%%%  %s' % (i, phrase[0], phrase[1]))
        
        if accuracy > 75:
            with open(f_name_wrong_answers, 'w') as f_wrong_answers:
                for phrase in wrong_answers:
                    f_wrong_answers.write(phrase[0] + ' %% ' + phrase[1] + '\n')


    def predict(self, question, return_lost_words=False):
        ''' Предварительная обработка вопроса к сети, перевод его в вектор, получение ответа от сети и перевод его в строку.
        1. question - строка с вопросом к сети
        2. return_lost_words - True, что бы вернуть список потерянных слов (которые отсутствуют в словаре и были удалены из результирующей последовательности)
        3. возвращает строку с ответом сети или, если return_lost_words=True, строку с ответом сети и list из потерянных слов '''
        
        if self.stp is None or self.w2v is None or self.model is None:
            print('[E] Сеть не загружена')
            return
        
        question = self.stp.prepare_question(question)
        if return_lost_words:
            question, lost_words = self.w2v.word2vec(question, True)
        else:
            question = self.w2v.word2vec(question)
        question = (question + 1.0) * 0.5
        answer = self.model.predict(question[np.newaxis,:])
        answer = answer * 2.0 - 1.0
        answer = self.w2v.vec2word(answer[0])
        answer = self.stp.prepare_answer(answer)
        if return_lost_words:
            return answer, lost_words
        else:
            return answer

# Субтитры (batch_size=32):
# При 250.000 обучающих примеров пиковое потребление оперативной памяти 19Гб (44Гб при сохранении результатов), обучение займёт примерно 112 часов
# При 100.000 обучающих примеров пиковое потребление оперативной памяти 9Гб, обучение займёт примерно 44 часа
# При 50.000 обучающих примеров пиковое потребление оперативной памяти 5.1Гб, обучение займёт примерно 22 часа
# При 25.000 обучающих примеров пиковое потребление оперативной памяти 3.4Гб, обучение займёт примерно 11 часов

# Глубина 3, запомнила примерно 450 ответов, ошибка 0.0443
# Глубина 4, на 7 итерации было 0.0647

# Глубина 1, 5000 примеров, циклов 100, эпох 5 - ошибка 0.0194 (270 из 950 правильных ответов)
# Глубина 1, 5000 примеров, циклов 200, эпох 5 - ошибка 0.0172 (311 из 880 правильных ответов)
# Глубина 2, 5000 примеров, циклов 500, эпох 5 - ошибка 0.0129 (2589 из 5000 правильных ответов)

# Пьесы (batch_size=32):
# Глубина 2, 1601 пример, 50 циклов, 5 эпох - ошибка 0.0217 (141 из 370 правильных ответов) (размерность вектора 500)
# Глубина 2, 1601 пример, 100 циклов, 5 эпох - ошибка 0.0117 (1105 из 1601 правильных ответов, точность 69.02%) (размерность вектора 500)
# Глубина 2, 1601 пример, 200 циклов, 5 эпох - ошибка 0.0104 (1207 из 1601 правильных ответов, точность 75.39%) (размерность вектора 500)
# Глубина 2, 1601 пример, 200 циклов, 5 эпох - ошибка 0.1111 (1512 из 1601 правильных ответов, точность 94.44%) (размерность вектора 500, эпох 100)
# Глубина 2, 1601 пример, 100 циклов, 5 эпох - ошибка 0.1131 (1493 из 1601 правильных ответов, точность 93.25%) (размерность вектора 500, эпох 100), batch_size=16
# Глубина 2, 1601 пример, 200 циклов, 5 эпох - ошибка 0.1900 (1588 из 1601 правильных ответов, точность 99.19%) (размерность вектора 500, эпох 500)
# Глубина 2, 1601 пример, 100 циклов, 5 эпох - ошибка 0.0274 (417 из 620 правильных ответов, точность 67.32%) (размерность вектора 300)
# Глубина 2, 1601 пример, 200 циклов, 5 эпох - ошибка 0.0247 (983 из 1410 правильных ответов, точность 69.72%) (размерность вектора 300)

# Диалоги (batch_size=32):
# Глубина 2, 5000 примеров, 500 циклов, 5 эпох - ошибка 0.0150 (2317 из 5000 правильных ответов, точность 46.34%)
# исправить ридми, настроить rhvoice
def main():
    f_name_plays = 'data/plays_ru/plays_ru.txt'
    f_name_subtitles = 'data/subtitles_ru/subtitles_ru.txt'
    f_name_conversations = 'data/conversations_ru/conversations_ru.txt'

    f_name_prepared_subtitles = 'data/subtitles_ru/prepared_subtitles_ru.pkl'

    f_name_enc_plays = 'data/plays_ru/encoded_plays_ru.npz'
    f_name_enc_subtitles = 'data/subtitles_ru/encoded_subtitles_ru.npz'
    f_name_enc_conversations = 'data/conversations_ru/encoded_conversations_ru.npz'
    
    f_name_w2v_model_plays = 'data/plays_ru/w2v_model_plays_ru.bin'
    f_name_w2v_model_subtitles = 'data/subtitles_ru/w2v_model_subtitles_ru.bin'
    f_name_w2v_model_conversations = 'data/conversations_ru/w2v_model_conversations_ru.bin'

    ttt = TextToText(train=True)
    ttt.prepare(f_name_plays, f_name_prepared_subtitles=f_name_prepared_subtitles, size=500, epochs=500, logging=True)
    #ttt.load_prepared(name_dataset='plays_ru')
    ttt.train(f_name_enc_plays, depth_model=2, training_cycles=200, epochs=5)

    new_ttt = TextToText(name_dataset='plays_ru')
    while(True):
        quest = input('Вы: ')
        answer = new_ttt.predict(quest)
        print('\t=> %s\n' % answer)


def on_stop(*args):
    print('\n[i] Остановлено')
    os._exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == 'Linux':
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    main()