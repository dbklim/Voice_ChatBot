#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Предварительная обработка исходных данных в формате "вопрос %% ответ" для обучения нейронной сети.
'''

import matplotlib.pyplot as plt
import numpy as np
import re
import pickle
import sys

import curses
import os


curses.setupterm()

# Создание папки для хранения данных, если она не была создана ранее и/или файл с исходными данными находится в другой папке
if not os.path.exists('data'):
    os.makedirs('data')


class SourceToPrepared:
    ''' Предназначен для преобразования пар "вопрос %% ответ" в последовательности фиксированного размера. Например:

    Вход: "Зачем нужен этот класс? %% Для подготовки данных"
    
    Выход: [['<PAD>', ..., '<PAD>', '?', 'класс', 'этот', 'нужен', 'Зачем', '<GO>'],
    ['Для', 'подготовки', 'данных', '<EOS>', '<PAD>', ..., '<PAD>']]

    Если объект класса используется только для преобразования предложений для работы с обученной сетью, то нужно задать max_sequence_length
    (максимальная длина последовательности (предложения) = размер входа сети). '''
    def __init__(self, max_sequence_length=None):
        if max_sequence_length is not None:
            print('[i] Установлена максимальная длина предложения %i слов(-а)' % max_sequence_length)
            self.max_sequence_length = max_sequence_length


    def prepare_all(self, f_name_source_data, f_name_training_sample=None):
        ''' Осуществляет преобразование пар "вопрос %% ответ" в последовательности фиксированного размера.
        1. f_name_source_data - имя входного .txt файла с исходными данными
        2. f_name_training_sample - имя выходного .pkl файла с предварительно обработанными парами [вопрос,ответ] (по умолчанию prepared_+f_name_source_data+.pkl) '''
        
        if f_name_training_sample is None:
            f_name_training_sample = f_name_source_data[:f_name_source_data.rfind('/')+1] + 'prepared_' + f_name_source_data[f_name_source_data.rfind('/')+1:]
            f_name_training_sample = f_name_training_sample.replace('.txt', '.pkl')

        dataset = self.__dataset_load(f_name_source_data)
        if len(dataset) < 5:
            print("\n[E] Исходный файл содержит меньше 10 пар 'вопрос %% ответ'\n")
            return
        dataset = self.__dataset_clean(dataset)
        dataset = self.__dataset_split(dataset)

        # Количество слов в каждом вопросе и ответе в каждой паре [вопрос, ответ]
        number_words = np.asarray([[len(q), len(a)] for [q, a] in dataset])
        self.max_sequence_length = number_words.max() + 2  # количество входов сети

        '''
        # Что бы посмотреть предложения, которые длиннее 25 слов
        for words in data:
            if len(words[0]) > 23:
                print('0: %s || %i' % (words[0], len(words[0])))
            if len(words[1]) > 23:
                print('1: %s || %i' % (words[1], len(words[1])))
        '''

        dataset = self.__fill_cells(dataset)
        self.__dataset_write(dataset, f_name_training_sample)
        
        f_name_histogram = f_name_source_data[:f_name_source_data.rfind('/')+1] + 'histogram_of_sizes_sentences_' + f_name_source_data[f_name_source_data.rfind('/')+1:]
        f_name_histogram = f_name_histogram.replace('_combined', '')
        f_name_histogram = f_name_histogram.replace('.txt', '.png')
        self.__dataset_info(number_words, f_name_histogram)


    def combine_subtitles(self, f_name_source_subtitles, f_name_subtitles=None):
        ''' Считывание очищенных субтитров из f_name_source_subtitles, объединение их в пары "вопрос %% ответ" (нечётные с чётными) и
        сохранение результата в f_name_subtitles_combined.
        1. f_name_source_subtitles - имя .txt файла с очищенными субтитрами из Taiga Corpus (подробнее в https://github.com/Desklop/Russian_subtitles_dataset)
        2. f_name_subtitles - имя .txt файла с объединёнными в пары субтитрами (по умолчанию f_name_source_subtitles без source_)
        
        !!!Выполняется очень долго!!! '''

        if f_name_subtitles is None:
            f_name_subtitles = f_name_source_subtitles[:f_name_source_subtitles.rfind('/')+1] + \
                                        f_name_source_subtitles[f_name_source_subtitles.rfind('/')+1:]
            f_name_subtitles = f_name_subtitles.replace('source_', '')

        print('[i] Считывание очищенных субтитров из %s' % f_name_source_subtitles)
        with open(f_name_source_subtitles, 'r') as f_ru_subtitles:
            subtitles = f_ru_subtitles.readlines()

        if subtitles[0].find('%%') != -1:
            print('[W] Субтитры уже объединены в пары')
            print('[i] Сохранение объединённых субтитров в %s' % f_name_subtitles)
            with open(f_name_subtitles, 'w') as f_ru_subtitles:
                for subtitle in subtitles:
                    f_ru_subtitles.write(subtitle)
            return

        if len(subtitles) % 2 != 0:
            del subtitles[-1]
        
        print('[i] Объединение субтитров в пары... ')
        i = 0
        while i < len(subtitles):
            if i % 1000 == 0 or i == len(subtitles) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Объединение субтитров в пары... %i из %i' % (i, len(subtitles)))
            subtitles[i] = subtitles[i][:len(subtitles[i])-1] # удаление \n
            subtitles[i] += ' %% ' + subtitles[i+1]
            del subtitles[i+1]
            i += 1
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Объединение субтитров в пары... %i из %i' % (len(subtitles), len(subtitles)))
        
        print('[i] Сохранение объединённых субтитров в %s' % f_name_subtitles)
        with open(f_name_subtitles, 'w') as f_ru_subtitles:
            for subtitle in subtitles:
                f_ru_subtitles.write(subtitle)


    def combine_conversations(self, f_name_source_conversations, f_name_conversations=None):
        ''' Считывание диалогов и реплик из f_name_source_conversations, объединение их в пары "вопрос %% ответ" (нечётные с чётными) и
        сохранение результата в f_name_conversations.
        1. f_name_source_conversations - имя .txt файла с диалогами (подробнее в https://github.com/Koziev/NLP_Datasets)
        2. f_name_conversations - имя .txt файла с объединёнными в пары диалогами (по умолчанию f_name_source_conversations без source_) '''
        
        if f_name_conversations is None:
            f_name_conversations = f_name_source_conversations[:f_name_source_conversations.rfind('/')+1] + \
                                            f_name_source_conversations[f_name_source_conversations.rfind('/')+1:]
            f_name_conversations = f_name_conversations.replace('source_', '')

        print('[i] Считывание набора диалогов из %s' % f_name_source_conversations)
        with open(f_name_source_conversations, 'r') as f_conversations:
            conversations = f_conversations.readlines()

        print('[i] Объединение диалогов в пары... ')
        i = 0
        while i < len(conversations):
            if i % 1000 == 0 or i == len(conversations) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Объединение диалогов в пары... %i из %i' % (i, len(conversations)))
            if conversations[i].find('-') == 0 and conversations[i+1].find('-') == 0:
                conversations[i] = conversations[i][:len(conversations[i])-1] # удаление \n
                conversations[i] = conversations[i][2:] # удаление '-' в начале строки
                conversations[i] += ' %% ' + conversations[i+1][2:]
                del conversations[i+1]
                i += 1
            elif conversations[i].find('-') == 0 and conversations[i+1] == '\n':
                temp = conversations[i-1].split(' %% ')
                temp[-1] = temp[-1][:len(temp[-1])-1] # удаление \n
                conversations[i] = temp[-1] + ' %% ' + conversations[i][2:]
                del conversations[i+1]
                i += 1
            else:
                del conversations[i]
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Объединение диалогов в пары... %i из %i' % (len(conversations), len(conversations)))

        print('[i] Сохранение объединённых диалогов в %s' % f_name_conversations)
        with open(f_name_conversations, 'w') as f_conversations_combined:
            for conversation in conversations:
                f_conversations_combined.write(conversation)


    def compress_conversations(self, f_name_conversations, f_name_conversations_comp=None):
        ''' Сжатие диалогов и реплик из f_name_conversations и сохранение результата в f_name_conversations_comp.
        1. f_name_conversations - имя .txt файла с диалогами (подробнее в https://github.com/Koziev/NLP_Datasets)
        2. f_name_conversations_comp - имя .txt файла с объединёнными в пары диалогами (по умолчанию f_name_conversations+_compress.txt) '''
        if f_name_conversations_comp is None:
            f_name_conversations_comp = f_name_conversations[:f_name_conversations.rfind('/')+1] + \
                                        f_name_conversations[f_name_conversations.rfind('/')+1:] + '_compress.txt'

        print('[i] Считывание набора диалогов из %s' % f_name_conversations)
        with open(f_name_conversations, 'r') as f_conversations:
            conversations = f_conversations.readlines()

        print('[i] Сжатие... ')
        i = 0
        number_delete = 0
        while i < len(conversations):
            if i % 1000 == 0 or i == len(conversations) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Сжатие... %i из %i' % (i, len(conversations)))
            # удалять по двум большим буквам
            temp = conversations[i]
            temp = temp.lower()
            temp = temp.replace('ё', 'е')
            if temp != '\n' and (temp.find('йе') != -1 or temp.find(' - ') != -1 or temp.find('"') != -1 or temp.find('мансур') != -1 \
               or temp.find('...') != -1 or temp.find('..') != -1):
                number_delete += 1
                del conversations[i]
            i += 1
            #elif conversations[i].find('-') == 0 and conversations[i+1] == '\n':
            #    temp = conversations[i-1].split(' %% ')
            #    temp[-1] = temp[-1][:len(temp[-1])-1] # удаление \n
            #    conversations[i] = temp[-1] + ' %% ' + conversations[i][2:]
            #    del conversations[i+1]
            #    i += 1
            #else:
            #    del conversations[i]
            

        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Сжатие... %i из %i' % (len(conversations), len(conversations)))

        if number_delete > 0:
            print('[i] Удалено %i элементов' % number_delete)

        print('[i] Удаление дубликатов...')
        print('[i] Исходный размер: %i' % len(conversations))
        conversations = list(set(conversations))
        print('[i] Новый размер: %i' % len(conversations))

        print('[i] Сохранение ужатых диалогов в %s' % f_name_conversations_comp)
        with open(f_name_conversations_comp, 'w') as f_conversations_combined:
            for conversation in conversations:
                f_conversations_combined.write(conversation)


    def prepare_question(self, question):
        ''' Предварительная обработка вопроса: удаление всего, что не является русскими буквами, разбиение на отдельные слова
        и приведение полученной последовательности к необходимой длине (max_sequence_length).
        1. question - строка с вопросом
        2. возращает преобразованную строку

        !!!Если не был вызван prepare_plays(), необходимо задать max_sequence_length!!! '''

        if self.max_sequence_length is None:
            print('[E] Перед использованием необходимо задать максимальную длину итоговой последовательности (предложения)!')
            return ['error']
        
        question = self.__clean_question(question)
        question = self.__dataset_split(question)
        question = self.__fill_cells_question(question)    
        return question


    def prepare_answer(self, answer):
        ''' Преобразует ответ сети в виде последовательности фиксированного размера в предложение.
        1. answer - последовательность фиксированного размера
        2. возвращает строку с ответом '''
        try:
            i = answer.index('<EOS>')
        except:
            i = len(answer)
        answer = ' '.join([ word for word in answer[0:i] if (word != '<PAD>') and (word != '<EOS>') and (word != '<GO>') ])       
        return self.__prepare_answer(answer)


    def get_questions(self, f_name_source_data):
        ''' Загрузка из файла текста в виде списка строк "вопрос %% ответ" и выделение вопросов.
        1. f_name_source_data - имя входного .txt файла с исходными данными
        2. возвращает список вопросов '''
        with open(f_name_source_data, 'r') as f_source_data: 
            dataset = f_source_data.readlines()

        # Удаление пробелов в начале и конце каждой строки "вопрос %% ответ" и разбиение каждой строки на две части
        dataset = [ pair.strip() for pair in dataset ]
        dataset = [ pair.split('%%') for pair in dataset ] 

        result = []
        for pair in dataset: # для всех пар [вопрос, ответ]
            question = pair[0].strip() # удаление лишних пробелов
            if len(question) > 0:
                result.append(question) # сохранение вопросов
        return result[:2000]


    def __prepare_answer(self, answer):
        ''' Очистка ответа от повторений знаков препинания и замена букв в нижнем регистре после знаков препинания '!', '?' и '.' 
        на буквы в верхнем регистре. '''

        # Удаление пробелов перед ',', '.', '!', '?' и '…'
        answer = re.sub(r'\s,', ',', answer)
        answer = re.sub(r'\s\.', '.', answer)
        answer = re.sub(r'\s!', '!', answer)
        answer = re.sub(r'\s\?', '?', answer)
        answer = re.sub(r'\s…', '…', answer)
        # Замена нескольких подряд идущих '.', ',', '!' и '?' на одиночные
        answer = re.sub(r'\.{2,5}', '…', answer)
        answer = re.sub(r',{2,5}', ',', answer)
        answer = re.sub(r'!{2,5}', '!', answer)
        answer = re.sub(r'\?{2,5}', '?', answer)
        # Замена конструкций вида '.,' и ',.' на '.' и ','
        answer = re.sub(r'\.,{1,5}', '.', answer)
        answer = re.sub(r'\.\?{1,5}', '?', answer)
        answer = re.sub(r',\?{1,5}', '?', answer)
        answer = re.sub(r'\.!{1,5}', '!', answer)
        answer = re.sub(r',!{1,5}', '!', answer)
        answer = re.sub(r',\.{1,5}', ',', answer)
        answer = re.sub(r',\?{1,5}', ',', answer)

        answer = re.sub(r'!,{1,5}', '!', answer)
        answer = re.sub(r'!\.{1,5}', '!', answer)
        answer = re.sub(r'\?,{1,5}', '?', answer)
        answer = re.sub(r'\?\.{1,5}', '?', answer)

        for_exit = True
        while for_exit: # замена маленькой буквы на большую после '!', '?', '.' и '…'
            if answer.find('! ') != -1 and answer[answer.find('! ')+2:][0].islower():
                answer = answer[:answer.find('! ')+2] + answer[answer.find('! ')+2:][0].upper() + answer[answer.find('! ')+3:]
            elif answer.find('? ') != -1 and answer[answer.find('? ')+2:][0].islower():
                answer = answer[:answer.find('? ')+2] + answer[answer.find('? ')+2:][0].upper() + answer[answer.find('? ')+3:]
            elif answer.find('. ') != -1 and answer[answer.find('. ')+2:][0].islower():
                answer = answer[:answer.find('. ')+2] + answer[answer.find('. ')+2:][0].upper() + answer[answer.find('. ')+3:]
            elif answer.find('… ') != -1 and answer[answer.find('… ')+2:][0].islower():
                answer = answer[:answer.find('… ')+2] + answer[answer.find('… ')+2:][0].upper() + answer[answer.find('… ')+3:]
            else:
                for_exit = False
        
        if len(answer) > 0 and (answer[:2] == '! ' or answer[:2] == '? ' or answer[:2] == '. ' or answer[:2] == ', ' or answer[:2] == '… '):
            answer = answer[2].lower() + answer[3:]
        if len(answer) > 0 and answer[-1] == ',':
            answer = answer[:len(answer)-1]
        while len(answer) > 0 and (answer[0] == '.' or answer[0] == ',' or answer[0] == '!' or answer[0] == '?' or answer[0] == '…'):
            answer = answer[1:]
        if len(answer) == 0:
            answer = '…'
        answer = answer.strip()
        return answer


    def __dataset_load(self, f_name_source_data):
        ''' Загрузка из файла и разбиение на пары текста в виде списка строк "вопрос %% ответ". '''
        print('[i] Загрузка исходных данных из %s' % f_name_source_data)
        with open(f_name_source_data, 'r') as f_source_data:
            dataset = f_source_data.readlines()
        print("[i] Считано %s строк 'вопрос %%%% ответ'" % len(dataset))

        # Удаление пробелов в начале и конце каждой строки "вопрос %% ответ" и разбиение каждой строки на две части
        dataset = [ pair.strip() for pair in dataset ]
        dataset = [ pair.split('%%') for pair in dataset ]
        
        print('[i] Разделение строк на пары [вопрос, ответ]...')
        i = 0
        number_error_elements = 0
        number_empty_elements = 0
        result = []
        for pair in dataset: # для всех пар [вопрос, ответ]
            if i % 1000 == 0 or i == len(dataset) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Разделение строк на пары [вопрос, ответ]... %i из %i' % (i, len(dataset)))
            if len(pair) != 2: # если пара не полная - ошибка
                number_error_elements += 1
                i += 1
                continue
            question = pair[0].strip()
            answer = pair[1].strip()
            if len(question) > 0 and len(answer) > 0:
                result.append([question, answer])
            else: # если один из пары содержит только пробелы - ошибка
                number_empty_elements += 1
            i += 1
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Разделение строк на пары [вопрос, ответ]... %i из %i' % (len(dataset), len(dataset)))

        if number_error_elements > 0:
            print("[W] %i строк не удовлетворяют условию 'вопрос %% ответ'" % number_error_elements)
        if number_empty_elements > 0:
            print('[W] %i строк содержат только пробелы' % number_empty_elements)
        return result


    def __dataset_clean(self, dataset):
        ''' Очистка всех пар [вопрос, ответ] от знаков препинания и неподдерживаемых символов. '''
        print('[i] Очистка всех вопросов и ответов...')
        i = 0
        number_empty_elements = 0
        result = []
        while i < len(dataset):
            if i % 1000 == 0 or i == len(dataset) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Очистка всех вопросов и ответов... %i из %i' % (i, len(dataset)))
            result.append([self.__clean_question(dataset[i][0]), self.__clean_answer(dataset[i][1])])
            if len(result[-1][0]) == 0 or len(result[-1][1]) == 0:
                number_empty_elements += 1
                del result[-1]
            i += 1
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Очистка всех вопросов и ответов... %i из %i' % (len(dataset), len(dataset)))
        if number_empty_elements > 0:
            print('[W] Найдено и удалено %i неполных пар [вопрос, ответ]' % number_empty_elements)
        return result


    def __clean_question(self, question):
        ''' Очистка вопроса от знаков препинания и неподдерживаемых символов. '''
        question = question.lower()
        question = re.sub(r'\.', '', question) 
        question = re.sub(r',', '', question) 
        question = re.sub(r':', '', question) 
        question = re.sub(r'-', ' ', question) 
        question = re.sub(r';', ' ', question) 
        question = re.sub(r'!', '', question) 
        question = re.sub(r'\?', '', question)
        question = re.sub(r'…', '', question)  
        question = re.sub(r'\.{2,5}', '', question)               
        #quest = re.sub(r'\.{2,5}', '...', quest)
        question = re.sub(r'"', '', question)
        question = re.sub(r"'", '', question)
        question = re.sub(r'«|»', '', question)
        question = re.sub(r'ё', 'е', question)
        question = re.sub(r'\([^()]*\)', ' ', question) # удаление скобок вместе с содержимым
        question = re.sub(r'\({1,5}|\){1,5}', ' ', question) # удаление отдельно стоящих скобок
        #quest = re.sub(r'\.\.\.*', '…', quest) 
        #quest = re.sub(r'[\W]+', '', quest) # удаление всех не букв
        return question
    

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


    def __dataset_split(self, dataset):
        ''' Разбиение пар [вопрос, ответ] или одиночного вопроса на отдельные слова (знаки препинания - отдельные элементы). 
        Максимальная длина предложения ограничена 28 словами. '''
        if isinstance(dataset, str): # если полученный объект - одиночный вопрос
            result = self.__tokenizer(dataset) # разбиение вопроса на слова
            result = [ word for word in reversed(result) ] # перестраивание слов в обратном порядке
        else:
            print('[i] Разбиение пар на отдельные слова...')
            i = 0
            number_empty_elements = 0
            number_large_elements = 0
            result = []
            while i < len(dataset):
                if i % 1000 == 0 or i == len(dataset) - 1:
                    os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                    print('[i] Разбиение пар на отдельные слова... %i из %i' % (i, len(dataset)))
                result.append([self.__tokenizer(dataset[i][0]), self.__tokenizer(dataset[i][1])]) # разбиение пар [вопрос, ответ] на слова
                result[-1] = [[word for word in reversed(result[-1][0])], result[-1][1]] # перестраивание слов вопроса в обратном порядке
                if len(result[-1][0]) > 28 or len(result[-1][1]) > 28:
                    number_large_elements += 1
                    del result[-1]
                elif len(result[-1][0]) == 0 or len(result[-1][1]) == 0:
                    number_empty_elements += 1
                    del result[-1]
                i += 1
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
            print('[i] Разбиение пар на отдельные слова... %i из %i' % (len(dataset), len(dataset)))
            if number_large_elements > 0:
                print('[W] Найдено и удалено %i слишком длинных пар [вопрос, ответ]' % number_large_elements)
            if number_empty_elements > 0:
                print('[W] Найдено и удалено %i неполных пар [вопрос, ответ]' % number_empty_elements)
        return result
    
    
    def __tokenizer(self, sentence):
        ''' Разбиение строки на слова. '''
        result = re.split(r'(\W)', sentence) # разбиение строки на последовательность из слов и знаков препинания
        result = [ word for word in result if word.strip() ] # удаление пустых элементов из последовательности
        if len(result) > 1:
            if result[-1] == '.': # удаление точки в конце, если она есть
                del result[-1]
        return result


    def __fill_cells(self, dataset):
        ''' Выравнивание всех пар по размеру. '''
        print('[i] Приведение предложений в парах к длине %i...' % self.max_sequence_length)
        i = 0
        result = []
        while i < len(dataset):
            if i % 1000 == 0 or i == len(dataset) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Приведение предложений в парах к длине %i... %i из %i' % (self.max_sequence_length, i, len(dataset)))
            result.append([self.__fill_cells_question(dataset[i][0]), self.__fill_cells_answer(dataset[i][1])])
            i += 1
        os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
        print('[i] Приведение предложений в парах к длине %i... %i из %i' % (self.max_sequence_length, len(dataset), len(dataset)))
        return result


    def __fill_cells_answer(self, answer):
        ''' Выравнивание ответа по размеру, заполняя пустые места словом <PAD>. Например: [..., '<EOS>', '<PAD>', ...] '''
        result = answer + ['<EOS>'] + ['<PAD>'] * (self.max_sequence_length - len(answer) - 1) 
        return result


    def __fill_cells_question(self, question):
        ''' Выравнивание вопроса по размеру, заполняя пустые места словом <PAD>. Например: [..., '<PAD>', 'вопрос', '<GO>'] '''
        result = ['<PAD>'] * (self.max_sequence_length - len(question) - 1) + question + ['<GO>']
        return result


    def __dataset_write(self, dataset, f_name_training_sample):
        ''' Запись полученного набора данных в .pkl файл. '''
        print('[i] Сохранение результата в %s' % f_name_training_sample)
        with open(f_name_training_sample, 'wb') as file:
            pickle.dump(dataset, file)


    def __dataset_info(self, number_words, f_name_histogram=None):
        ''' Вывод информации о полученных парах [вопрос, ответ], построение гистограммы размеров предложений и сохранение её в f_name_histogram
        (по умолчанию data/histogram_of_sizes_sentences.png). '''
        if f_name_histogram is None:
            f_name_histogram = 'data/histogram_of_sizes_sentences.png'

        print('[i] Размеры предложений в парах [вопрос, ответ]:')
        print('\tмаксимальный: %i, %i' % (number_words.max(axis=0)[0], number_words.max(axis=0)[1])) # максимальная длинна
        print('\tминимальный: %i, %i' % (number_words.min(axis=0)[0], number_words.min(axis=0)[1])) # минимальная длинна
        print('\tмeдиана: %i, %i' % (np.median(number_words, axis=0).astype(int)[0], np.median(number_words, axis=0).astype(int)[1])) # медианная длинна

        # Гистограмма размеров предложений
        print('[i] Построение гистограммы размеров предложений...')
        plt.figure()
        plt.hist(x=number_words, label=['вопросы', 'ответы'])
        plt.title('Гистограмма размеров предложений') 
        plt.ylabel('количество')
        plt.xlabel('длинна')
        plt.legend()
        plt.savefig(f_name_histogram, dpi=100)


# Пиковое потребление оперативной памяти:
# При обработке субтитров - 6.8Гб
# При обработке диалогов - 250Мб
# При обработке пьес - 60Мб

def main():
    f_name_plays = 'data/plays_ru/plays_ru.txt'
    f_name_subtitles = 'data/subtitles_ru/subtitles_ru.txt'
    f_name_conversations = 'data/conversations_ru/conversations_ru.txt'

    f_name_source_subtitles = 'data/subtitles_ru/source_subtitles_ru.txt'
    f_name_source_conversations = 'data/conversations_ru/source_conversations_ru.txt'

    stp = SourceToPrepared()
    #stp.combine_subtitles(f_name_source_subtitles)
    #stp.combine_conversations(f_name_source_conversations)
    #stp.prepare_all(f_name_plays)
    #stp.prepare_all(f_name_subtitles)
    #stp.prepare_all(f_name_conversations)
    stp.compress_conversations(f_name_source_conversations)

    #temp = stp.get_questions(f_name_plays)

    while True:
        sentence = input('Введите предложение: ')
        result = stp.prepare_question(sentence)
        print(result)


if __name__ == '__main__':
    main()