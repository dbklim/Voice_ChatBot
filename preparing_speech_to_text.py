#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Создание статической языковой модели и фонетического словаря для PocketSphinx на основе
вопросов из обучающей выборки. Для создания необходимо вызвать функцию building_language_model().
Используется CMUclmtk_v0.7 и text2dict из ru4sphinx.
'''

import os
import sys
import re
import subprocess
from itertools import groupby

# Создание временной папки, если она была удалена
if not os.path.exists('temp'):
    os.makedirs('temp')

# Попробовать использовать CMUclmtk через pip3 install python-cmuclmtk
# https://github.com/Holzhaus/python-cmuclmtk 

class LanguageModel:
    ''' Предназначен для создания статической языковой модели и фонетического словаря для PocketSphinx на основе вопросов из обучающей выборки. '''
    def build_language_model(self, f_name_source_data, len_source_data=None):
        ''' Создание статической языковой модели и фонетического словаря для PocketSphinx на основе вопросов из обучающей выборки.
        После выполнения в .../pocketsphinx/model/ будет создана языковая модель ru_bot_+name_dataset+.lm и фонетический словарь ru_bot_+name_dataset+.dic
        Так же языковая модель и фонетический словарь сохраняются в temp/prepared_questions.lm и temp/prepared_questions.dic
        1. f_name_source_data - имя .txt файла с обучающей выборкой
        2. len_source_data - если None: использовать весь f_name_source_data, иначе - первые len_source_data элементов
        Для работы используется text2wfreq, wfreq2vocab, text2idngram и idngram2lm из CMUclmtk_v0.7.
            
        !!! Включено ограничение на 20000 слов в словаре, что бы изменить, нужно в коде заменить wfreq2vocab на wfreq2vocab -top 20000
        [20000 - необходимый предел размера словаря] !!! '''

        print('[i] Создание статической языковой модели и фонетического словаря для pocketsphinx')        
        current_dirname = os.path.dirname(os.path.realpath(__file__)) + '/temp/'

        # Подготовка данных
        result = self.__preparing_questions(f_name_source_data, len_source_data)
        name_dataset = f_name_source_data[f_name_source_data.rfind('/')+1:f_name_source_data.rfind('.txt')]
        if result is not None and result == 'error':
            return
        
        f_name_log = 'temp/building_language_model_' + name_dataset + '.log'
        f_name_prepared_questions = current_dirname + 'prepared_questions_' + name_dataset + '.txt'
        f_name_prepared_questions_vocab = current_dirname + 'prepared_questions_' + name_dataset + '_vocab.txt'
        f_name_prepared_questions_idngram = current_dirname + 'prepared_questions_' + name_dataset + '.idngram'
        f_name_prepared_questions_lm = current_dirname + 'prepared_questions_' + name_dataset + '.lm'
        f_name_prepared_questions_dic = current_dirname + 'prepared_questions_' + name_dataset + '.dic'
        f_name_final_lm = 'ru_bot_' + name_dataset + '.lm'
        f_name_final_dic = 'ru_bot_' + name_dataset + '.dic'
        
        # Включено ограничение на 20000 слов в словаре, что бы изменить, нужно заменить wfreq2vocab на wfreq2vocab -top 20000
        print('[i] Построение словаря из вопросов обучающей выборки... ', end='')
        # text2wfreq <prepared_questions.txt | wfreq2vocab> prepared_questions.tmp.vocab.txt
        command_line = "text2wfreq <'" + f_name_prepared_questions + "' | wfreq2vocab> '" + f_name_prepared_questions_vocab + "'"
        proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out = out.decode()
        err = err.decode()
        log = out + err + '\n===========================================================================================\n\n'
        if log.find('text2wfreq : Done.') != -1 and log.find('wfreq2vocab : Done.') != -1:
            print('ок')
        else:
            print('\n[E] Ошибка, подробнее в %s' % f_name_log)
            with open(f_name_log, 'w') as f_log:
                f_log.write(log)
            return

        print('[i] Построение idngram... ', end='')
        # text2idngram -vocab prepared_questions.vocab.txt -idngram prepared_questions.idngram < prepared_questions.txt
        command_line = "text2idngram -vocab '" + f_name_prepared_questions_vocab + "' -idngram '" + \
                       f_name_prepared_questions_idngram + "' < '" + f_name_prepared_questions + "'"
        proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out = out.decode()
        err = err.decode()
        log += out + err + '\n===========================================================================================\n\n'
        if log.find('text2idngram : Done.') != -1:
            print('ок')
        else:
            print('\n[E] Ошибка, подробнее в %s' % f_name_log)
            with open(f_name_log, 'w') as f_log:
                f_log.write(log)
            return

        print('[i] Построение статической языковой модели... ', end='')
        # idngram2lm -vocab_type 0 -idngram prepared_questions.idngram -vocab prepared_questions.vocab.txt -arpa prepared_questions.lm
        command_line = "idngram2lm -vocab_type 0 -idngram '" + f_name_prepared_questions_idngram + "' -vocab '" + \
                       f_name_prepared_questions_vocab + "' -arpa '" + f_name_prepared_questions_lm + "'"
        proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out = out.decode()
        err = err.decode()
        log += out + err
        if log.find('idngram2lm : Done.') != -1:
            print('ок')
        else:
            print('\n[E] Ошибка, подробнее в %s' % f_name_log)
            with open(f_name_log, 'w') as f_log:
                f_log.write(log)
            return

        # Сохранение логов
        with open(f_name_log, 'w') as f_log:
            f_log.write(log)
        
        # Удаление первых 6 строк из словаря, 4 из которых содержат комментарий от CMUclmtk, а оставшиеся 2 - <s> и </s>
        vocabulary = []
        with open(f_name_prepared_questions_vocab, 'r') as f_vocab:
            vocabulary = f_vocab.readlines()
        for i in range(5, -1, -1):
            del vocabulary[i]
        with open(f_name_prepared_questions_vocab, 'w') as f_vocab:
            for word in vocabulary:
                f_vocab.write(word)

        # Создание фонетического словаря из обычного, созданного text2wfreq и wfreq2vocab
        self.create_dictionary(f_name_prepared_questions_vocab, f_name_prepared_questions_dic)
        
        # Копирование языковой модели ru_bot.lm и фонетического словаря ru_bot.dic в ...pythonx.x/dist-packages/pocketsphinx/model/
        print('[i] Копирование языковой модели ru_bot_' + name_dataset + '.lm и фонетического словаря ru_bot_' + name_dataset + \
              '.dic в ...python' + str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '/dist-packages/pocketsphinx/model/')
        path_dict_for_ps = '/usr/local/lib/python' + str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '/dist-packages/pocketsphinx/model/'
        command_line = "sudo cp '" + f_name_prepared_questions_lm + "' '" + path_dict_for_ps + f_name_final_lm + "'"
        subprocess.call(command_line, shell=True)

        command_line = "sudo cp '" + f_name_prepared_questions_dic + "' '" + path_dict_for_ps + f_name_final_dic + "'"
        subprocess.call(command_line, shell=True)
        print('[i] Готово')


    def __preparing_questions(self, f_name_source_data, len_source_data=None):
        ''' Чтение пар вопрос %% ответ из f_name_source_data, выделение вопросов, удаление знаков препинания и неподдерживаемых
        символов, заключение каждого предложения в <s>..</s> и сохранение результата в temp/prepared_questions_+f_name_source_data
        1. f_name_source_data - имя .txt файла с обучающей выборкой
        2. len_source_data - если None: использовать весь f_name_source_data, иначе - первые len_source_data элементов '''

        print('[i] Подготовка вопросов...')
        if not os.path.isfile(f_name_source_data):
            print("\n[E] Файл '" + f_name_source_data + "' не существует\n")
            return 'error'
        
        dataset = self.__dataset_load(f_name_source_data)
        if len_source_data is not None and len_source_data <= len(dataset):
            dataset = dataset[:len_source_data]
        print('[i] Считано %s пар из %s' % (len(dataset), f_name_source_data))
        dataset = self.__dataset_clean(dataset)

        dataset = [re.sub(r'\s{2,10}', ' ', q) for q in dataset]

        for i in range(len(dataset)):
            dataset[i] = '<s> ' + dataset[i] + ' </s>'
            i += 1

        dataset = [q for q, _ in groupby(dataset)]

        name_dataset = f_name_source_data[f_name_source_data.rfind('/')+1:f_name_source_data.rfind('.txt')]
        with open('temp/prepared_questions_' + name_dataset + '.txt', 'w') as file:
            for q in dataset:
                file.write(q + '\n')


    def __dataset_load(self, f_name_source_data):
        ''' Загрузка из файла текста в виде списка строк "вопрос %% ответ" и выделение вопросов. '''
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
        return result


    def __dataset_clean(self, dataset):
        ''' Очистка всех вопросов от знаков препинания и неподдерживаемых символов. '''
        result = [ self.__clean_question(q) for q in dataset ]
        return result


    def __clean_question(self, question):
        ''' Очистка вопроса от знаков препинания и неподдерживаемых символов. '''
        question = question.lower()
        question = re.sub(r'\.', '', question) 
        question = re.sub(r',', '', question) 
        question = re.sub(r':', '', question) 
        question = re.sub(r'–', '-', question)
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


    def create_dictionary(self, f_name_vocab, f_name_dic):
        ''' Создание фонетического словаря из обычного, созданного text2wfreq и wfreq2vocab из вопросов обучающей выборки. 
        После выполнения будет создан f_name_dic.
        1. f_name_vocab - имя .txt файла обычного словаря
        2. f_name_dic - имя .dic файла фонетического словаря
        Для работы используется text2dict из https://github.com/zamiron/ru4sphinx/tree/master/text2dict '''
        
        print('[i] Создание фонетического словаря...\n')
        current_dirname = os.path.dirname(os.path.realpath(__file__))

        # perl dict2transcript.pl prepared_questions_vocab.txt prepared_questions.dic
        command_line = "perl '" + current_dirname + "/text_to_dict/dict_to_transcript.pl' '" + f_name_vocab + "' '" + f_name_dic + "'"
        subprocess.call(command_line, shell=True)
        print()




def main():
    f_name_plays = 'data/plays_ru/plays_ru.txt'
    f_name_subtitles = 'data/subtitles_ru/subtitles_ru.txt'
    f_name_conversations = 'data/conversations_ru/conversations_ru.txt'

    lm = LanguageModel()
    lm.build_language_model(f_name_plays)


if __name__ == '__main__':
    main()