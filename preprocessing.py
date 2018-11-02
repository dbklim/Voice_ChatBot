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

        '''
        # Что бы посмотреть предложения, которые длиннее 25 слов
        for words in data:
            if len(words[0]) > 23:
                print('0: %s || %i' % (words[0], len(words[0])))
            if len(words[1]) > 23:
                print('1: %s || %i' % (words[1], len(words[1])))
                '''

        data = self.__fill_cells(data)
                
        print('[i] Сохранение результата в %s и %s.pkl' % (filename_out, filename_out[:filename_out.find('.')]))
        self.__data_write(data, filename_out) 
        self.__data_info(number_words)


    def prepare_quest(self, quest):
        ''' Преобразует вопрос к сети в последовательность фиксированного размера.
        1. quest - строка, содержащая вопрос
        2. возращает преобразованную строку

        !!!Если не был вызван prepare_all(), необходимо задать input_size!!! '''

        quest = self.__clean_quest(quest)
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
        answer = ' '.join([ w for w in answer[0:i] if (w != '<PAD>') and (w != '<EOS>') and (w != '<GO>') ])       
        return self.__prepare_answer(answer)


    def get_questions(self, filename_in):
        ''' Загрузка из файла текста в виде списка строк "вопрос %% ответ" и выделение вопросов.
        1. filename_in - имя входного .txt файла с исходными данными
        2. возвращает список вопросов '''
        with open(filename_in, 'r') as file: 
            content = file.readlines()

        # удаление пробелов в начале и конце каждой строки "вопрос %% ответ" и разбиение каждой строки на две части
        content = [ str.strip() for str in content ]
        content = [ str.split('%%') for str in content ] 

        result = []
        for x in content: # для всех пар [вопрос, ответ]
            q = x[0].strip() # удаление лишних пробелов
            if len(q) > 0:
                result.append(q) # сохранение вопросов
        return result


    def __prepare_answer(self, answer):
        ''' Очистка ответа от повторений знаков препинания и замена букв в нижнем регистре после знаков препинания '!', '?' и '.' 
        на буквы в верхнем регистре. '''

        # Удаление пробелов перед ',', '.', '!', '?' и '…'
        answer = re.sub(r'\s,', ',', answer)
        answer = re.sub(r'\s\.', '.', answer)
        answer = re.sub(r'\s!', '!', answer)
        answer = re.sub(r'\s\?', '?', answer)
        answer = re.sub(r'\s…', '…', answer)
        # Замена нескольких подряд идущих ','  '!' и '?' на одиночные
        answer = re.sub(r',{2,5}', ',', answer)
        answer = re.sub(r'!{2,5}', '!', answer)
        answer = re.sub(r'\?{2,5}', '?', answer)
        # Замена конструкций вида '.,' и ',.' на '.' и ','
        answer = re.sub(r'\.,{1,5}', '.', answer)
        answer = re.sub(r'\.\?{1,5}', '?', answer)
        answer = re.sub(r',\.{1,5}', ',', answer)
        answer = re.sub(r',\?{1,5}', ',', answer)

        current_string = answer
        final_answer = ''
        i = 0
        while i < len(current_string)-2:  # Перебор каждых двух подряд идущих символов    
            # Если текущие два подряд идущих символа совпадают с каким-либо шаблоном в условии
            if current_string[i:i+2] == '! ' or current_string[i:i+2] == '? ' or current_string[i:i+2] == '. ':
                final_answer += current_string[0:i+2] # Сохраняем в финальный текст подстроку до шаблона и сам шаблон
                current_string = current_string[i+2:len(current_string)] # Оставляем только подстроку, идущую после шаблона
                if current_string[0].islower(): # Если первый символ подстроки после шаблона в нижнем регистре - замена на верхний регистр
                    current_string = current_string[:0] + current_string[0].upper() + current_string[1:]
                i = 0 
                continue
            i += 1
        if len(current_string) > 2:
            final_answer += current_string
        if len(final_answer) == 0:
            final_answer = current_string
        return final_answer


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
        ''' Очистка всех пар [вопрос, ответ] от знаков препинания и неподдерживаемых символов. '''
        result = [ [ self.__clean_quest(q), self.__clean_answer(a) ] for [q,a] in data ]
        return result


    def __clean_quest(self, quest):
        ''' Очистка вопроса от знаков препинания и неподдерживаемых символов. '''
        quest = quest.lower()
        quest = re.sub(r'\.', '', quest) 
        quest = re.sub(r',', '', quest) 
        quest = re.sub(r':', '', quest) 
        quest = re.sub(r'-', ' ', quest) 
        quest = re.sub(r';', ' ', quest) 
        quest = re.sub(r'!', '', quest) 
        quest = re.sub(r'\?', '', quest)
        quest = re.sub(r'…', '', quest)  
        quest = re.sub(r'\.{2,5}', '', quest)               
        #quest = re.sub(r'\.{2,5}', '...', quest)
        quest = re.sub(r'"', '', quest)
        quest = re.sub(r"'", '', quest)
        quest = re.sub(r'«|»', '', quest)
        quest = re.sub(r'ё', 'е', quest)
        quest = re.sub(r'\([^()]*\)', ' ', quest) # удаление скобок вместе с содержимым
        quest = re.sub(r'\({1,5}|\){1,5}', ' ', quest) # удаление отдельно стоящих скобок
        #quest = re.sub(r'\.\.\.*', '…', quest) 
        #quest = re.sub(r'[\W]+', '', quest) # удаление всех не букв
        return quest
    

    def __clean_answer(self, answer):
        ''' Очистка ответа от лишних знаков препинания и неподдерживаемых символов. '''
        answer = answer.lower()
        #answer = re.sub(r'\.', '', answer) 
        #answer = re.sub(r',', '', answer) 
        answer = re.sub(r':', '', answer) 
        #answer = re.sub(r'-', '', answer) 
        answer = re.sub(r';', ',', answer) 
        #answer = re.sub(r'!', '', answer)         
        #answer = re.sub(r'…', '', answer)    
        #answer = re.sub(r'\.{2,5}', '', answer)     
        answer = re.sub(r'\.{2,5}', '…', answer)
        answer = re.sub(r'"', '', answer)
        answer = re.sub(r"'", '', answer)
        answer = re.sub(r'«|»', '', answer)
        answer = re.sub(r'ё', 'е', answer)
        answer = re.sub(r'\([^()]*\)', '', answer) # удаление скобок вместе с содержимым
        answer = re.sub(r'\({1,5}|\){1,5}', '', answer) # удаление отдельно стоящих скобок
        #answer = re.sub(r'\.\.\.*', '…', answer) 
        #answer = re.sub(r'[\W]+', '', answer) # удаление всех не букв
        return answer


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
        if len(result) > 1:
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
    #b.prepare_all('data/source_data.txt', 'data/prepared_data.txt')
    a = b.get_questions('data/source_data.txt')

    c = {'questions':a}
    print(c.get('questions')[1600])
    print()


if __name__ == '__main__':
    main()