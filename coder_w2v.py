# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для кодирования подготовленных пар [вопрос, ответ] в векторную форму 
и обратно. Используется кодировщик word2vec из библиотеки gensim.
'''

import sys
import numpy as np
from gensim.models import word2vec
import logging as log
import pickle
import multiprocessing


class CoderW2V:
    ''' Кодирует последовательности слов (предложений) в вектор и наоборот. Используется кодировщик word2vec из библиотеки gensim. 
    
    Если объект класса используется только для кодирования/декодирования вопросов к сети и ответов сети, то
    необходимо указать mode = 'load_model' и имя файла с моделью filename_w2v_model. В противном случае ничего указывать
    не нужно. '''
    def __init__(self, mode = None, filename_w2v_model = None):
        if mode == 'load_model':
            self.model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename_w2v_model, binary = True)

    def words2vec(self, filename_in, filename_out, filename_w2v_model, filename_w2v_vocab, filename_w2v_neighborhood, 
                  size, epochs, window = 5, logging = False):
        ''' Осуществляет кодирование предложений в вектора. Предназначен для кодирования обучающей выборки для seq2seq модели НС.

        1. filename_in - имя входного .pkl файла с предварительно обработанными парами [вопрос,ответ]
        2. filename_out - имя выходного .npz файла с векторным представлением слов в парах [вопрос,ответ]
        3. filename_w2v_model - имя .bin файла для сохранения обученной W2V модели
        4. filename_w2v_vocab - имя .txt файла для сохранения полученного словаря W2V
        5. filename_w2v_neighborhood - имя .txt файла для сохранения найденных слов-соседей для каждого слова в словаре
        3. size - размер окна обработки вектора (что равно числу входов модели W2V), которое подаётся на вход модели W2V.
        Обычно используют значения от десятков до сотен. Большие значения требуют больше данных обучения, но могут привести 
        к более точным моделям.
        4. epochs - число эпох обучения модели W2V
        5. window - максимальное расстояние между текущим и прогнозируемым словом в предложении
        6. logging - включение вывода данных в процессе обучения модели W2V
        
        Входные предложения должны иметь вид, например: [['<PAD>', ..., '<PAD>', '?', 'класс', 'этот', 'нужен', 'Зачем', '<GO>'], 
        ['Для', 'кодирования', 'предложений', '<EOS>', '<PAD>', ..., '<PAD>']]'''
        
        print('[i] Загрузка данных из %s' % filename_in)
        with open(filename_in, 'rb') as file:
            data = pickle.load(file)

        print('\tколичество: %i пар' % len(data))
        print('\tдлинна строки: %i слов' % len(data[0][0]))

        vocabulary = set([ w for p in data for s in p for w in s ])
        print("\tразмер словаря: %i слов" % len(vocabulary))

        print('[i] Обучение W2V...')
        self.model = self.__w2v_fit(data, filename_w2v_model, size, window, epochs, logging)
        
        print('[i] Проверка результата обучения W2V...')
        self.__model_test(self.model, data, filename_w2v_vocab, filename_w2v_neighborhood)

        print('[i] Кодировка и сортировка строк...')
        Q, A = self.__data_w2v_encode(self.model, data)

        print('[i] Сохранение учебного набора в %s' % filename_out)
        np.savez(filename_out, Q = Q, A = A)

    def word2vec(self, quest):
        ''' Кодирует последовательность фиксированного размера в вектор. 
        1. quest - последовательность фиксированного размера
        2. возвращает вектор 
        В случае, если какое-либо слово отсутствует в словаре, возвращается строка вида 'Error:неизвестное_слово'.  '''
        result = []
        for w in quest:
            try:
                result.append(self.model[w])
            except KeyError:
                return 'Error:' + w
        # result = np.asarray([self.model[w] for w in quest])
        return np.asarray(result)

    def vec2word(self, answer):
        ''' Декодирует вектор в последовательность фиксированного размера. 
        1. answer - ответ сети в виде вектора
        2. возвращает последовательность фиксированного размера '''
        answ_seq = [ self.model.similar_by_vector(v)[0][0] for v in answer ]
        return answ_seq

    def __w2v_fit(self, data, filename_w2v_model, w_size, window_size, number_epochs, logging):
        ''' Обучение модели W2V. Возвращает обученную модель.'''      
        if logging == True:
            log.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = log.INFO)

        sentences = [ q + a for [q,a] in data ] 
        model = word2vec.Word2Vec(min_count = 1, size = w_size, window = window_size, workers = multiprocessing.cpu_count())
        model.build_vocab(sentences)
        model.train(sentences, epochs = number_epochs, total_examples = len(data))

        model.wv.save_word2vec_format(filename_w2v_model, binary = True) 
        return model

    def __model_test(self, model, data, filename_w2v_vocabulary, filename_w2v_neighborhood):
        ''' Тестирование обученной модели: определение размера и соранение словаря, определение числа потерянных слов и 
        сохранение найденных слов-соседей для каждого слова в словаре. '''

        voc2 = [ w for w in model.wv.vocab.keys() ]
        print('\tразмер словаря W2V: %i слов' % len(voc2))
        with open(filename_w2v_vocabulary, 'w') as file:
            for w in voc2:
                print(w, file = file)

        vocabulary = set([ w for p in data for s in p for w in s ])

        i = 0 # количество потерянных слов
        with open(filename_w2v_neighborhood, 'w') as file:
            for w in vocabulary:
                try:                    
                    words_neighbors = model.wv.most_similar(positive = [w], topn = 4) # ближайшие по W2V слова к слову w
                    r = [ s for s,d in words_neighbors ] 
                    print(w, r, file = file)
                except:
                    print('[W] Слово "%s" отсутствует в словаре W2V.' % w)
                    i += 1
        if i > 0:
            print('[i] Потерянных слов %i из %i.' % (i, len(vocabulary)))

    def __data_w2v_encode(self, model, data):
        ''' Кодирование предложений из data в вектор. Возвращает отдельно массивы закодированных вопросов и ответов. '''
        Q = np.asarray([ [ model[w] for w in q ] for q,a in data ])
        A = np.asarray([ [ model[w] for w in a ] for q,a in data ])
        return Q, A
    



def main():
    coder = CoderW2V()

    f_in = 'data/prepared_data.pkl'
    f_out = 'data/encoded_data.npz'
    f_model = 'data/w2v_model.bin'
    f_vocab = 'data/w2v_vocabulary.txt'
    f_nbhd = 'data/w2v_neighborhood.txt'

    coder.words2vec(f_in, f_out, f_model, f_vocab, f_nbhd, size = 150, epochs = 100) # было 500 1000 для авторских данных


if __name__ == '__main__':
    print("[i] Количество CPU: %i" % multiprocessing.cpu_count() )
    main()