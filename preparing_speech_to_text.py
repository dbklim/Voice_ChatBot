# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для создания статической языковой модели и фонетического словаря для PocketSphinx на основе вопросов из обучающей выборки 
для НС seq2seq. Для создания необходимо вызвать функцию building_language_model(). Используется CMUclmtk_v0.7 и text2dict из ru4sphinx.
'''

import os
import sys
import re
import subprocess
from itertools import groupby


def data_load(filename):
    ''' Загрузка из файла текста в виде списка строк "вопрос %% ответ" и выделение вопросов. '''
    with open(filename,'r') as file: 
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

def data_clean(data):
    ''' Очистка всех вопросов от знаков препинания и неподдерживаемых символов. '''
    result = [ clean_quest(q) for q in data ]
    return result

def clean_quest(quest):
    ''' Очистка вопроса от знаков препинания и неподдерживаемых символов. '''
    quest = quest.lower()
    quest = re.sub(r'\.', '', quest) 
    quest = re.sub(r',', '', quest) 
    quest = re.sub(r':', '', quest) 
    quest = re.sub(r'–', '-', quest)
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

def preparing_questions(filename_in):
    ''' Чтение пар вопрос %% ответ из filename_in, выделение вопросов, удаление знаков препинания 
    и неподдерживаемых символов, заключение каждого предложения в <s>..</s> и сохранение результата 
    в temp/prepared_questions.txt '''

    print('[i] Подготовка вопросов...')
    filename_in = 'data/source_data.txt'
    data = data_load(filename_in)
    print('[i] Считано %s пар из %s' % (len(data), filename_in))
    data = data_clean(data)

    data = [re.sub(r'\s{2,10}', ' ', q) for q in data]

    i = 0
    for i in range(len(data)):
        data[i] = '<s> ' + data[i] + ' </s>'
        i += 1

    data = [q for q, _ in groupby(data)]

    with open('temp/prepared_questions.txt', 'w') as file:
        for d in data:
            file.write(d + '\n')

def building_language_model(filename_source_data):
    ''' Создания статической языковой модели и фонетического словаря для PocketSphinx на основе вопросов из обучающей выборки.
    После выполнения в .../pocketsphinx/model/ будет создана языковая модель ru_bot.lm и фонетический словарь ru_bot.dic
    1. filename_source_data - имя .txt файла с обучающей выборкой
    Для работы используется text2wfreq, wfreq2vocab, text2idngram и idngram2lm из CMUclmtk_v0.7.
        
    !!! Включено ограничение на 20000 слов в словаре, что бы изменить, нужно в коде заменить wfreq2vocab на wfreq2vocab -top 20000
    [20000 - необходимый предел размера словаря] !!!'''

    print('[i] Создание статической языковой модели и фонетического словаря для pocketsphinx')

    filename_log = 'temp/building_language_model.log'
    current_dirname = os.path.dirname(os.path.realpath(__file__)) + '/temp/'    

    # Подготовка данных
    preparing_questions(filename_source_data)

    # Построение словаря из вопросов обучающей выборки. Включено ограничение на 20000 слов в словаре, что бы изменить, 
    # нужно заменить wfreq2vocab на wfreq2vocab -top 20000
    print('[i] Построение словаря из вопросов обучающей выборки... ', end = '')
    # text2wfreq <prepared_questions.txt | wfreq2vocab> prepared_questions.tmp.vocab.txt
    command_line = "text2wfreq <'" + current_dirname + "prepared_questions.txt' | wfreq2vocab> '" + current_dirname + "prepared_questions.tmp.vocab.txt'"
    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    err = err.decode()
    log = out + err + '\n===========================================================================================\n\n'
    if log.find('text2wfreq : Done.') != -1 and log.find('wfreq2vocab : Done.') != -1:
        print('ок')
    else:
        print('\n[E] Ошибка, подробнее в %s' % filename_log)
        return

    command_line = "cp '" + current_dirname + "prepared_questions.tmp.vocab.txt' '" + current_dirname + "prepared_questions.vocab.txt'"
    subprocess.call(command_line, shell=True)

    # Построение статической языковой модели
    print('[i] Построение статической языковой модели... ', end = '')
    # text2idngram -vocab prepared_questions.vocab.txt -idngram prepared_questions.idngram < prepared_questions.txt
    command_line = "text2idngram -vocab '" + current_dirname + "prepared_questions.vocab.txt' -idngram '" + current_dirname + "prepared_questions.idngram' < '" + current_dirname + "prepared_questions.txt'"
    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    err = err.decode()
    log += out + err + '\n===========================================================================================\n\n'
    
    # idngram2lm -vocab_type 0 -idngram prepared_questions.idngram -vocab prepared_questions.vocab.txt -arpa prepared_questions.lm
    command_line = "idngram2lm -vocab_type 0 -idngram '" + current_dirname + "prepared_questions.idngram' -vocab '" + current_dirname + "prepared_questions.vocab.txt' -arpa '" + current_dirname + "prepared_questions.lm'"
    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    err = err.decode()
    log += out + err
    if log.find('text2idngram : Done.') != -1 and log.find('idngram2lm : Done.') != -1:
        print('ок')
    else:
        print('\n[E] Ошибка, подробнее в %s' % filename_log)
        return

    # Сохранение логов
    with open(filename_log, 'w') as file:
        file.write(log)
    
    # Удаление первых 6 строк из словаря, 4 из которых содержат комментарий от CMUclmtk, а последние 2 - <s> и </s>
    vocab = []
    with open('temp/prepared_questions.vocab.txt', 'r') as file:
        vocab = file.readlines()
    for i in range(5, -1, -1):
        del vocab[i]
    with open('temp/prepared_questions.vocab.txt', 'w') as file:
        for word in vocab:
            file.write(word)

    # Создание фонетического словаря из обычного, созданного text2wfreq и wfreq2vocab
    create_dictionary('temp/prepared_questions.vocab.txt')
    
    # Копирование языковой модели ru_bot.lm и фонетического словаря ru_bot.dic в .../pocketsphinx/model/
    print('[i] Копирование языковой модели ru_bot.lm и фонетического словаря ru_bot.dic в .../pocketsphinx/model/')
    path_dict_for_ps = '/usr/local/lib/python3.5/dist-packages/pocketsphinx/model/'
    command_line = "sudo cp '" + current_dirname + "prepared_questions.lm' '" + path_dict_for_ps + "ru_bot.lm'"
    subprocess.call(command_line, shell=True)

    command_line = "sudo cp '" + current_dirname + "prepared_questions.dic' '" + path_dict_for_ps + "ru_bot.dic'"
    subprocess.call(command_line, shell=True)
    print('[i] Готово')

def create_dictionary(filename_vocabulary):
    ''' Создание фонетического словаря из обычного, созданного text2wfreq и wfreq2vocab из вопросов обучающей выборки. 
    После выполнения будет создан prepared_questions.dic.
    1. filename_vocabulary - имя словаря 
    Для работы используется text2dict из https://github.com/zamiron/ru4sphinx/tree/master/text2dict '''
    
    print('[i] Создание фонетического словаря...\n')

    current_dirname = os.path.dirname(os.path.realpath(__file__))
    filename_vocabulary = os.path.dirname(os.path.realpath(__file__)) + '/' + filename_vocabulary

    # perl dict2transcript.pl prepared_questions.vocab.txt prepared_questions.dic
    command_line = "perl '" + current_dirname + "/text2dict/dict2transcript.pl' '" + filename_vocabulary + "' '" + current_dirname + "/temp/prepared_questions.dic'"
    subprocess.call(command_line, shell=True)
    print()




def main():   
    building_language_model('data/source_data.txt')


if __name__ == '__main__':
    main()