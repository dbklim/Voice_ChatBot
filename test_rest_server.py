#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or 18.04
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''
Тестирование rest_server.py путём отправки на него всех поддерживаемых запросов.
'''

import time
import logging
import base64
import signal
import os
import sys
import curses
import subprocess
import http.client as http_client
import requests
from pydub import AudioSegment


def train():  
    print('[i] Обучение сети пока не поддерживается сервером.')


def predict(speech_recognition=False, speech_synthesis=False):
    ''' Работа с обученной моделью seq2seq.
    1. speech_recognition - включение распознавания речи с микрофона с помощью PocketSphinx
    2. speech_synthesis - включение озвучивания ответов с помощью RHVoice '''

    # Что бы записать звук с микрофона в терминале: arecord output.wav (для остановки надо нажать Ctrl+C)
    # Что бы проиграть .wav в терминале: aplay output.wav

    #name_dataset = f_name_w2v_model_plays[f_name_w2v_model_plays.rfind('w2v_model_')+len('w2v_model_'):f_name_w2v_model_plays.rfind('.bin')]
    #ttt = TextToText(f_name_w2v_model=f_name_w2v_model, f_name_model=f_name_model, f_name_model_weights=f_name_model_weights)

    if speech_recognition:
        print('[i] Загрузка языковой модели для распознавания речи...')
        #stt = SpeechToText('from_microphone', name_dataset)

    if speech_synthesis:
        print('[i] Загрузка синтезатора речи...')
        #tts = TextToSpeech('playback')

    print()
    question = ''
    while(True):
        if speech_recognition:
            print('Слушаю... (для остановки нажмите Ctrl+C)')
            command_line = "arecord voice.wav"
            subprocess.call(command_line, shell=True)
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
            print('Пользователь: ' + quest)
        else:
            question = input('Вы: ')
        '''
        answer, lost_words = ttt.predict(question, True)
        print('\t=> %s' % answer)
        if len(lost_words) > 0:
            print('[w] Потерянные слова: ' + ', '.join(lost_words) + '\n')
        else:
            print()
        if speech_synthesis:
            tts.get(answer)
        '''



def get_address_on_local_network():
    ''' Определение адреса машины в локальной сети с помощью утилиты 'ifconfig' из пакета net-tools.
    1. возвращает строку с адресом или 127.0.0.1, если локальный адрес начинается не с 192.Х.Х.Х или 172.Х.Х.Х
    
    Проверено в Ubuntu 16.04 и 18.04. '''
    
    command_line = 'ifconfig'
    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    if out.find('not found') != -1:
        print("\n[E] 'ifconfig' не найден.")
        sys.exit(0)
    if out.find('inet 127.0.0.1') != -1:
        template = 'inet '
    elif out.find('inet addr:127.0.0.1') != -1:
        template = 'inet addr:'
    i = 0
    host_192xxx = None
    host_172xxx = None
    while host_192xxx is None or host_172xxx is None: 
        out = out[out.find(template) + len(template):]
        host = out[:out.find(' ')]
        out = out[out.find(' '):]
        if host.find('192.168') != -1:
            host_192xxx = host
        elif host.find('172.') != -1:
            host_172xxx = host
        i += 1
        if i >= 10:
            break
    if host_192xxx:
        return host_192xxx
    elif host_172xxx:
        return host_172xxx
    else:
        print('\n[E] Неподдерживаемый формат локального адреса, требуется корректировка исходного кода.\n')
        return '127.0.0.1'


def main():
    '''
    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger('requests.packages.urllib3')
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True
    '''

    auth = base64.b64encode('bot:test_bot'.encode())
    headers = {'Authorization' : 'Basic ' + auth.decode()}
    http = 'http://'
    addr = '127.0.0.1:5000'
    addr = '192.168.2.205:5000'
    addr = '172.17.0.2:5000'
    addr = get_address_on_local_network() + ':5000'

    '''
    curses.setupterm()
    quest = ''
    print('Записываю... (для остановки нажмите Enter)')
    command_line = "temp/arecord voice.wav"
    proc = subprocess.Popen(command_line, stdout=sys.stdout, shell=True, preexec_fn=os.setsid)
    quest = input()
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
    time.sleep(0.01)
    os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
    print('                               ')
    os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
    print('Распознано: ' + quest)
    '''

    print('GET-запрос на /chatbot/about')
    result = requests.get(http + addr + '/chatbot/about', headers=headers, verify=False)
    data = result.json()
    data = data.get('text')
    print('Результат:')
    print(data + '\n')
    print(result.headers)
    print(result.text)

    print('GET-запрос на /chatbot/questions')
    result = requests.get(http + addr + '/chatbot/questions', headers=headers, verify=False)
    data = result.json()
    data = data.get('text')
    print('Результат (первые 5 элементов):')
    print(str(data[:5]) + '\n')
    print(result.headers)
    print(result.text[:1000] + '\n')
    '''
    print('POST-запрос на /chatbot/speech-to-text')
    print('Чтение temp/speech.opus...')
    data = None
    with open('temp/speech.opus', 'rb') as audio:
        data = audio.read()
    data = base64.b64encode(data)
    data = {'opus' : data.decode()}
    result = requests.post(http + addr + '/chatbot/speech-to-text', headers=headers, json=data, verify=False)
    data = result.json()
    data = data.get('text')
    print('Результат:')
    print(data + '\n')
    print(result.headers)
    print(result.text)
'''
    print('POST-запрос на /chatbot/speech-to-text')
    print('Чтение temp/test1.wav...')
    with open('temp/test1.wav', 'rb') as audio:
        data = audio.read()
    data = base64.b64encode(data)
    data = {'wav' : data.decode()}
    result = requests.post(http + addr + '/chatbot/speech-to-text', headers=headers, json=data, verify=False)
    data = result.json()
    data = data.get('text')
    print('Результат:')
    print(data + '\n')
    print(result.headers)
    print(result.text)
    
    print('POST-запрос на /chatbot/text-to-speech')
    print('Отправлено: как тебя зовут?')
    data = {'text':'как тебя зовут?'}
    result = requests.post(http + addr + '/chatbot/text-to-speech', headers=headers, json=data, verify=False)
    data = result.json()
    data = base64.b64decode(data.get('wav'))
    print('Результат в temp/synthesized_speech.wav\n')
    with open('temp/synthesized_speech.wav', 'wb') as audio:
        audio.write(data)
    print(result.headers)
    print(result.text[:1000] + '\n')
    '''
    print('POST-запрос на /chatbot/speech-to-text')
    print('Чтение temp/synthesized_speech.wav...')
    with open('temp/synthesized_speech.wav', 'rb') as audio:
        data = audio.read()
    data = base64.b64encode(data)
    data = {'wav' : data.decode()}
    result = requests.post(http + addr + '/chatbot/speech-to-text', headers=headers, json=data, verify=False)
    data = result.json()
    data = data.get('text')
    print('Результат:')
    print(data + '\n')
    print(result.headers)
    print(result.text + '\n')
    '''
    print('POST-запрос на /chatbot/text-to-text')
    data = {'text':'как тебя зовут?'}
    result = requests.post(http + addr + '/chatbot/text-to-text', headers=headers, json=data, verify=False)
    data = result.json()
    data = data.get('text')
    print('Результат:')
    print(data + '\n')
    print(result.headers)
    print(result.text + '\n')

    while True:
        question = input('Введите вопрос боту: ')
        #question = 'мне нужна помощь'
        start_time = time.time()
        data = {'text':question}
        result = requests.post(http + addr + '/chatbot/text-to-text', headers=headers, json=data, verify=False)
        data = result.json()
        answer = data.get('text')
        print('Ответ: ' + answer)
        print('Время: ' + str(time.time() - start_time) + ' сек' + '\n')
        #time.sleep(0.2)
    
    '''
    curses.setupterm()
    print('[i] Выберите вариант удалённой работы с ботом:')
    print('\t1. train - обучение модели seq2seq')
    print('\t2. predict - работа с обученной моделью seq2seq')
    print('\t3. predict -ss - включено озвучивание ответов с помощью RHVoice')
    print('\t4. predict -sr - включено распознавание речи с помощью PocketSphinx')
    print('\t5. predict -ss -sr - включено озвучивание ответов и распознавание речи')
    while True:
        choice = input('Введите цифру: ')
        if choice == '1':
            choice = input('Вы уверены?(д/н) ')
            if choice == 'д':
                train()
            break
        elif choice == '2':
            predict()
            break
        elif choice == '3':
            predict(False, True)
            break
        elif choice == '4':
            predict(True, False)
            break
        elif choice == '5':
            predict(True, True)
            break
        else:
            os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
    '''



if __name__ == '__main__':
    main()