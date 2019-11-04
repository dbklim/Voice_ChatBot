#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Перевод подготовленных пар [вопрос, ответ] в векторную форму и обратно. Используется
кодировщик word2vec из библиотеки gensim.
'''

import pickle
import multiprocessing
import os
import sys
import curses
import logging as log
import numpy as np
from gensim.models import word2vec


curses.setupterm()


class WordToVec:
    ''' Позволяет переводить последовательности слов (предложений) в вектора и наоборот. Если объект класса используется только для
    кодирования предложений для работы с обученной сетью, то нужно указать имя файла с .bin моделью f_name_w2v_model. '''
    def __init__(self, f_name_w2v_model=None):
        if f_name_w2v_model is not None:
            print('[i] Загрузка модели word2vec из %s...' % f_name_w2v_model)
            self.model = word2vec.Word2VecKeyedVectors.load_word2vec_format(f_name_w2v_model, binary=True)
            #self.model.init_sims(replace=True) # Оптимизация, что бы модель word2vec занимала в оперативной памяти меньше места
            vocabulary = [ w for w in self.model.wv.vocab.keys() ]
            print('[i] Размер словаря word2vec: %i слов' % len(vocabulary))


    def build_word2vec(self, f_name_training_sample, f_name_prepared_subtitles=None, f_name_enc_training_sample=None, f_name_w2v_model=None, 
                       f_name_w2v_vocab=None, len_encode=None, size=500, min_count_repeat=1, window=5, epochs=None, logging=False):
        ''' Предназначен для построения модели word2vec и кодирования обучающей выборки в вектора.
        1. f_name_training_sample - имя входного .pkl файла с предварительно обработанными парами [вопрос,ответ]
        2. f_name_prepared_subtitles - имя .pkl файла с предварительно обработанными субтитрами (для расширения словаря модели word2vec)
        3. f_name_enc_training_sample - имя выходного .npz файла с векторным представлением слов в парах [вопрос,ответ] (по умолчанию encoded_+f_name_training_sample+.npz)
        4. f_name_w2v_model - имя .bin файла для сохранения обученной модели word2vec (по умолчанию w2v_model_+f_name_training_sample+.bin)
        5. f_name_w2v_vocab - имя .txt файла для сохранения словаря word2vec (по умолчанию w2v_vocabulary_+f_name_training_sample+.txt)
        6. len_encode - если None: закодировать весь f_name_training_sample, иначе - первые len_encode элементов
        7. size - рразмерность вектора, которым кодируется одно слово
        Обычно используют значения в несколько сотен, большие значения требуют больше данных обучения, но могут привести к более точным моделям.
        8. min_count - минимальное количество повторений (частота повторений) слова, что бы оно было включено в словарь
        9. window - максимальное расстояние между текущим и прогнозируемым словом в предложении
        10. epochs - число эпох обучения модели word2vec (если None, используется внутренний итератор)
        11. logging - включение вывода данных в процессе обучения модели word2vec
        Входные предложения должны иметь вид, например: [['<PAD>', ..., '<PAD>', '?', 'класс', 'этот', 'нужен', 'Зачем', '<GO>'], 
        ['Для', 'кодирования', 'предложений', '<EOS>', '<PAD>', ..., '<PAD>']] '''
        
        if f_name_enc_training_sample is None:
            f_name_enc_training_sample = f_name_training_sample.replace('prepared_', 'encoded_')
            f_name_enc_training_sample = f_name_enc_training_sample.replace('.pkl', '.npz')
        if f_name_w2v_model is None:
            f_name_w2v_model = f_name_training_sample.replace('prepared_', 'w2v_model_')
            f_name_w2v_model = f_name_w2v_model.replace('.pkl', '.bin')
        if f_name_w2v_vocab is None:
            f_name_w2v_vocab = f_name_training_sample.replace('prepared_', 'w2v_vocabulary_')
            f_name_w2v_vocab = f_name_w2v_vocab.replace('.pkl', '.txt')

        if f_name_prepared_subtitles is not None:
            print('[i] Загрузка субтитров из %s' % f_name_prepared_subtitles)
            with open(f_name_prepared_subtitles, 'rb') as f_subtitles:
                subtitles = pickle.load(f_subtitles)

            print('\tколичество предложений: %i пар' % len(subtitles))
            print('\tдлинна предложений: %i слов' % len(subtitles[0][0]))

        print('[i] Загрузка данных из %s' % f_name_training_sample)
        with open(f_name_training_sample, 'rb') as f_training_sample:
            training_sample = pickle.load(f_training_sample)

        print('\tколичество предложений: %i пар' % len(training_sample))
        print('\tдлинна предложений: %i слов' % len(training_sample[0][0]))
        
        if f_name_prepared_subtitles is not None:
            print('[i] Объединение...')
            dataset = subtitles + training_sample
        else:
            dataset = training_sample

        print('[i] Подсчёт размера полного словаря...')
        i = 0
        vocabulary = []
        for pair in dataset:
            if i % 1000 == 0 or i == len(dataset) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Подсчёт размера полного словаря... %i из %i' % (i, len(dataset)))
            for sentence in pair:
                for word in sentence:
                    vocabulary.append(word)
            i += 1
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Подсчёт размера полного словаря... %i из %i' % (len(dataset), len(dataset)))
        vocabulary = set(vocabulary)
        print('\tпредполагаемый размер полного словаря: %i слов' % len(vocabulary))
        
        self.__w2v_fit(dataset, f_name_w2v_model, min_count_repeat, size, window, epochs, logging)
        dataset = training_sample

        vocabulary = [ w for w in self.model.wv.vocab.keys() ]
        print('[i] Размер словаря word2vec: %i слов' % len(vocabulary))
        with open(f_name_w2v_vocab, 'w') as f_w2v_vocab:
            for word in vocabulary:
                print(word, file=f_w2v_vocab)

        # Оптимизация, что бы модель word2vec занимала в оперативной памяти меньше места (почему-то искажает декодировку векторов в слова)
        #self.model = self.model.wv
        #self.model.init_sims(replace=True)

        self.data_w2v_encode(dataset, f_name_enc_training_sample, len_encode)


    def word2vec(self, sentence, return_lost_words=False):
        ''' Кодирует последовательность фиксированного размера в вектор. 
        1. sentence - последовательность фиксированного размера
        2. return_lost_words - True, что бы вернуть список потерянных слов (которые отсутствуют в словаре и были удалены из результирующей последовательности)
        2. возвращает вектор в виде array или, если return_lost_words=True, вектор в виде array и list из потерянных слов '''

        result = []
        lost_words = []
        for word in sentence:
            try:
                result.append(self.model[word])
            except KeyError:
                lost_words.append(word)
                result.insert(0, self.model['<PAD>'])
        if return_lost_words:
            return np.array(result), lost_words
        else:
            return np.array(result)


    def vec2word(self, answer):
        ''' Декодирует вектор в последовательность фиксированного размера. 
        1. answer - ответ сети в виде вектора
        2. возвращает последовательность фиксированного размера '''
        answ_seq = [ self.model.similar_by_vector(v)[0][0] for v in answer ]
        return answ_seq


    def __w2v_fit(self, dataset, f_name_w2v_model, min_count_repeat, v_size, window_size, epochs, logging):
        ''' Обучение модели word2vec и сохранение полученной модели в f_name_w2v_model.
        1. dataset - list из преложений, каждое из которых является list из слов (один элемент list[i][j] = одно слово)
        2. f_name_w2v_model - имя .bin файла для сохранения обученной модели word2vec
        3. min_count_repeat - минимальное количество повторений (частота повторений) слова, что бы оно было включено в словарь
        4. v_size - размерность вектора, которым кодируется одно слово
        5. window_size - максимальное расстояние между текущим и прогнозируемым словом в предложении
        6. epochs - число эпох обучения модели word2vec (если None, используется внутренний итератор)
        7. logging - вывод данных в процессе обучения модели word2vec '''

        if logging:
            log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=log.INFO)
        print('[i] Обучение модели word2vec...')

        '''
        # Удаление слов-наполнителей, слов-маркеров и объединение вопросов с ответами
        clear_dataset = []
        for pair in dataset:
            clear_dataset.append([])
            for word in pair[0]:                
                if not (word in '<PAD>' or word in '<GO>' or word in '<EOS>'):
                    clear_dataset[-1].append(word)
            for word in pair[1]:                
                if not (word in '<PAD>' or word in '<GO>' or word in '<EOS>'):
                    clear_dataset[-1].append(word)
        clear_dataset.append(['<PAD>', '<GO>', '<EOS>'])
        '''

        sentences = [ q + a for [q,a] in dataset ]
        self.model = word2vec.Word2Vec(min_count=min_count_repeat, size=v_size, window=window_size, workers=multiprocessing.cpu_count())
        # Используется модель skip-gram (результат нулевой)
        #model = word2vec.Word2Vec(min_count=min_count_repeat, size=w_size, window=window_size, workers=multiprocessing.cpu_count(), sg=1)
        self.model.build_vocab(sentences)

        if epochs is None:
            epochs = self.model.iter
        self.model.train(sentences, epochs=epochs, total_examples=len(sentences))

        self.model.wv.save_word2vec_format(f_name_w2v_model, binary=True)


    def data_w2v_encode(self, dataset, f_name_enc_training_sample, len_enc_training_sample=None):
        ''' Кодирование предложений из dataset в вектор.
        1. dataset - предварительно обработанные пары [вопрос,ответ]
        2. f_name_enc_training_sample - имя выходного .npz файла с векторным представлением слов в парах [вопрос,ответ]
        3. len_enc_training_sample - если None: закодировать весь dataset, иначе - первые len_enc_training_sample элементов '''
        
        print('[i] Кодирование обучающей выборки...')
        questions = [ q for [q, a] in dataset ]
        answers = [ a for [q, a] in dataset ]

        if len_enc_training_sample is None or len_enc_training_sample > len(dataset):
            len_enc_training_sample = len(dataset)

        i = 0
        lost_words = []
        while i < len_enc_training_sample:
            if i % 1000 == 0 or i == len(questions) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Кодирование обучающей выборки... %i из %i' % (i, len(questions)))
            current_sentence = []
            for word in questions[i]:
                try:
                    current_sentence.append(self.model[word])
                except KeyError:
                    lost_words.append(word)
                    current_sentence.insert(0, self.model['<PAD>'])
            questions[i] = np.asarray(current_sentence)

            current_sentence = []
            for word in answers[i]:
                try:
                    current_sentence.append(self.model[word])
                except KeyError:
                    lost_words.append(word)
                    current_sentence.insert(-1, self.model['<PAD>'])
            answers[i] = np.asarray(current_sentence)
            i += 1
        questions = questions[:len_enc_training_sample]
        answers = answers[:len_enc_training_sample]
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Кодирование обучающей выборки... %i из %i    ' % (len(questions), len(questions)))

        if len(lost_words) > 0:
            print('[i] Количество потерянных слов: %i' % len(lost_words))
        
        print('[i] Сохранение закодированной обучащей выборки в %s' % f_name_enc_training_sample)
        np.savez_compressed(f_name_enc_training_sample, questions=questions, answers=answers)


    def w2v_test(self, test_word):
        ''' Тестирование обученной модели: поиск похожих на test_word слов в словаре.
        1. test_word - строка со словом
        2. возвращает list из 10 ближайших слов или 'not_found' если слова нет в словаре '''

        try:
            return self.model.most_similar([test_word])
        except KeyError:
            return 'not_found'


# Пиковое потребление оперативной памяти при кодировании субтитров в вектора (во время обучения - 8Гб):
# 55Гб (700.000 пар, размерность вектора 300)
# 52Гб (650.000 пар, размерность вектора 300)
# 42Гб (500.000 пар, размерность вектора 300)
# 25Гб (250.000 пар, размерность вектора 300), вес закодированных данных 18Гб (в сжатом виде 4Гб)
# 16Гб (100.000 пар, размерность вектора 300), вес закодированных данных 7.2Гб (в сжатом виде 1.6Гб)
# 11Гб (50.000 пар, размерность вектора 300), вес закодированных данных 3.6Гб (в сжатом виде 800Мб)
# 8Гб (25.000 пар, размерность вектора 300), вес закодированных данных 1.8Гб (в сжатом виде 400Мб)
# 7.5Гб (10.000 пар, размерность вектора 500), вес закодированных данных 1.2Гб (в сжатом виде 285Мб)
# 6.7Гб (5.000 пар, размерность вектора 500), вес закодированных данных 0.6Гб (в сжатом виде 143Мб)

# Пиковое потребление оперативной памяти при кодировании пьес в вектора (модель пьесы + субтитры, во время обучения 8.1Гб):
# 8.1Гб (1601 пар, размерность вектора 500)

# Пиковое потребление оперативной памяти при кодировании диалогов в вектора (модель диалоги + субтитры, во время обучения 6.6Гб):
# 24.2Гб (136.493 пар, размерность вектора 500)
# 3.0Гб (10.000 пар, размерность вектора 500)
# 860Мб (5.000 пар, размерность вектора 300)

def main():
    f_name_prepared_plays = 'data/plays_ru/prepared_plays_ru.pkl'
    f_name_prepared_conversations = 'data/conversations_ru/prepared_conversations_ru.pkl'
    f_name_prepared_subtitles = 'data/subtitles_ru/prepared_subtitles_ru.pkl'

    f_name_w2v_model_plays = 'data/plays_ru/w2v_model_plays_ru.bin'
    f_name_w2v_model_conversations = 'data/conversations_ru/w2v_model_conversations_ru.bin'
    f_name_w2v_model_subtitles = 'data/subtitles_ru/w2v_model_subtitles_ru.bin'

    f_name_enc_plays = 'data/plays_ru/encoded_plays_ru.npz'
    f_name_enc_subtitles = 'data/subtitles_ru/encoded_subtitles_ru.npz'
    f_name_enc_conversations = 'data/conversations_ru/encoded_conversations_ru.npz'

    # Обучающая выборка ограничена 5.000 примерами, т.к. в противном случае обучение на моей видеокарте nvidia gtx1070 займёт больше суток

    w2v = WordToVec()
    #w2v.build_word2vec(f_name_prepared_conversations, f_name_prepared_subtitles, len_encode=5000, size=500, epochs=None, logging=True)
    # size=500 или 1000 для набора данных из 1500-1600 обучающих пар, если size=300 - не удаётся обучить сеть с точностью >60-70%

    new_w2v = WordToVec(f_name_w2v_model_subtitles)

    #print('[i] Загрузка данных из %s' % f_name_prepared_conversations)
    #with open(f_name_prepared_conversations, 'rb') as f_training_sample:
    #    dataset = pickle.load(f_training_sample)
    #new_w2v.data_w2v_encode(dataset, f_name_enc_conversations, 10000)

    while True:
        word = input('Введите слово: ')
        words_neighbors = new_w2v.w2v_test(word)
        if isinstance(words_neighbors, str):
            print('Слово не найдено')
            continue
        result = [ s for s,d in words_neighbors ] 
        print(result)


if __name__ == '__main__':
    print('[i] Количество CPU: %i' % multiprocessing.cpu_count() )
    main()