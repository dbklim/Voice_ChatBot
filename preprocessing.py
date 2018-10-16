# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для предварительной обработки исходных данных для обучения.
'''

import matplotlib.pyplot as plt
import numpy as np
import re
import pickle
import sys


class Preparation:
    ''' Преобразует пары "вопрос %% ответ" в последовательности фиксированного размера.

    Например: 
    
    Вход: "Зачем нужен этот класс? %% Для подготовки данных"
    
    Выход: [['<PAD>', ..., '<PAD>', '?', 'класс', 'этот', 'нужен', 'Зачем', '<GO>'], 
    ['Для', 'подготовки', 'данных', '<EOS>', '<PAD>', ..., '<PAD>']] '''
    def prepare_all(self, filename_in, filename_out):
        ''' Осуществляет преобразование пар "вопрос %% ответ" в последовательности фиксированного размера. 
        1. filename_in - имя входного .txt файла с исходными данными
        2. filename_out - имя выходного .txt файла с преобразованными данными '''
        
        data = self.__data_load(filename_in)
        print('[i] Считано %s пар из %s' % (len(data), filename_in))

        data = self.__data_clean(data)
        data = self.__data_split(data)

        # колчиество слов в каждом вопросе и ответе в каждой паре [вопрос, ответ]
        number_words = np.asarray([ [ len(q), len(a) ] for [q,a] in data ]) 
        self.input_size = number_words.max() + 2  # количество входов сети

        data = self.__fill_cells(data)
                
        print('[i] Сохранение результата в %s и %s.pkl' % (filename_out, filename_out[:filename_out.find('.')]))
        self.__data_write(data, filename_out) 
        self.__data_info(number_words)


    def prepare_quest(self, quest):
        ''' Преобразует вопрос к сети в последовательность фиксированного размера.
        1. quest - строка, содержащая вопрос
        2. возращает преобразованную строку

        !!!Если не был вызван prepare_all(), необходимо задать input_size!!! '''

        quest = self.__clean_string(quest)
        quest = self.__data_split(quest)
        quest = self.__fill_cells_quest(quest)    
        return quest


    def prepare_answer(self, answer):
        ''' Преобразует ответ сети в виде последовательности фиксированного размера в предложение.
        1. answer - последовательность фиксированного размера
        2. возвращает строку с ответом '''
        try:
            i = answer.index('<EOS>')
        except:
            i = len(answer)
        return ' '.join([ w for w in answer[0:i] if (w != '<PAD>') and (w != '<EOS>') and (w != '<GO>') ])


    def __data_load(self, filename):
        ''' Загрузка из файла и разбиение на пары текста в виде списка строк "вопрос %% ответ". '''
        with open(filename,'r') as file: 
            content = file.readlines()

        # удаление пробелов в начале и конце каждой строки "запрос %% ответ" и разбиение 
        # каждой строки на две части
        content = [ str.strip() for str in content ]
        content = [ str.split('%%') for str in content ] 

        i = 1 # номер текущей строки
        result = []
        for x in content: # для всех пар [запрос,ответ]
            if len(x) != 2: # если пара не полная - ошибка
                print('[E] Строка %i' % i)
                return []
            # удаляем лишние пробелы
            q = x[0].strip() 
            a = x[1].strip()
            if len(q) > 0 and len(a) > 0:
                result.append([q,a])
            else: # если один из пары содержит только пробелы - ошибка
                print('[E] Строка %i' % i)
                return []
            i += 1
        return result


    def __data_clean(self, data):
        ''' Очистка всех пар [вопрос, ответ] от неподдерживаемых символов. '''
        result = [ [ self.__clean_string(q), self.__clean_string(a) ] for [q,a] in data ]
        return result
    

    def __clean_string(self, str):
        ''' Очистка одной строки от неподдерживаемых символов. '''
        result = str.lower()
        result = re.sub(r'\.', ' ', result) 
        #result = re.sub(r',', ' ', result) 
        result = re.sub(r':', ' ', result) 
        result = re.sub(r'-', ' ', result) 
        result = re.sub(r';', ' ', result) 
        result = re.sub(r'!', ' ', result) 
        result = re.sub(r'…', ' ', result) 
        result = re.sub(r'\.{1,5}', ' ', result)
        result = re.sub(r'"', ' ', result)
        #result = re.sub(r'\.\.\.*', '…', result) 
        #result = re.sub(r'[\W]+', ' ', result) # удаление всех не букв
        return result


    def __data_split(self, data):
        ''' Разбиение пар [вопрос, ответ] или одиночного вопроса на отдельные слова (знаки препинания - отдельные элементы). '''
        if isinstance(data, str): # если полученный объект - одиночный вопрос
            result = self.__tokenizer(data) # разбиение вопроса на слова
            result = [ w for w in reversed(result) ] # перестраивание слов в обратном порядке
        else:
            result = [ [ self.__tokenizer(q), self.__tokenizer(a) ] for [q,a] in data ] # разбиение пар [вопрос, ответ] на слова
            result = [ [ [ w for w in reversed(q) ], a ] for [q,a] in result ] # перестраивание слов вопроса в обратном порядке
        return result
    

    def __tokenizer(self, str):
        ''' Разбиение строки на слова. '''
        result = re.split(r'(\W)', str) # разбиение строки на последовательность из слов и знаков препинания
        result = [ word for word in result if word.strip() ] # удаление пустых элементов из последовательности
        if result[-1] == '.': # удаление точки в конце, если она есть
            del result[-1]
        return result


    def __fill_cells(self, data):
        ''' Выравнивание всех пар по размеру. '''
        result = [ [self.__fill_cells_quest(q), self.__fill_cells_answ(a)] for [q,a] in data ]
        return result


    def __fill_cells_answ(self, a):
        ''' Выравнивание ответа по размеру, заполняя пустые места словом <PAD>. Например: [..., '<EOS>', '<PAD>', ...] '''
        result = a + ['<EOS>'] + ['<PAD>'] * (self.input_size - len(a) - 1) 
        return result


    def __fill_cells_quest(self, q):
        ''' Выравнивание вопроса по размеру, заполняя пустые места словом <PAD>. Например: [..., '<PAD>', 'вопрос', '<GO>'] '''
        result = ['<PAD>'] * (self.input_size - len(q) - 1) + q + ['<GO>']
        return result


    def __data_write(self, data, filename):
        ''' Запись полученных данных в .txt и .pkl файлы. '''
        other_filename = filename[:filename.find('.')] + '.pkl'
        with open(other_filename, 'wb') as file:
            pickle.dump(data, file)
        with open(filename, 'w') as file:
            for d in data:
                print(d, file=file)


    def __data_info(self, number_words):
        ''' Вывод информации о полученных парах [вопрос, ответ]. '''
        print('[i] Размеры пар последовательностей:')
        print('\tмаксимальный: ', number_words.max(axis=0)) # максимальная длинна
        print('\tминимальный: ', number_words.min(axis=0)) # минимальная длинна
        print('\tмeдиана: ', np.median(number_words, axis=0).astype(int)) # медианная длинна

        # гистограмма размеров последовательностей
        print('[i] Построение гистограммы размеров последовательностей...')
        plt.figure()
        plt.hist(x = number_words, label = ['вопросы', 'ответы'])
        plt.title('Гистограмма размеров последовательностей') 
        plt.ylabel('количество')
        plt.xlabel('длинна')
        plt.legend()
        plt.savefig('data/features_hist.png', dpi=100)




def main():
    b = Preparation()
    b.prepare_all('data/source_data.txt', 'data/prepared_data.txt')

if __name__ == '__main__':
    main()