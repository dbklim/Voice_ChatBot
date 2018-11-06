#!/usr/bin/python3
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
REST-сервер для взаимодействия с ботом. Используется Flask и WSGIServer.
'''

import os
import sys
import signal
import platform
import base64
import json
import subprocess
import socket
from logging.config import dictConfig
from datetime import datetime
from functools import wraps

from flask import Flask, redirect, jsonify, abort, request, make_response, __version__ as flask_version
from flask_httpauth import HTTPBasicAuth
from gevent import __version__ as wsgi_version
from gevent.pywsgi import WSGIServer 

from tensorflow import get_default_graph

from preprocessing import Preparation
from prediction import Prediction
from text_to_speech import tts
from speech_to_text import SpeechRecognition            


# Удаление старых логов
if os.path.exists('server.log'):
    os.remove('server.log')
    for i in range(1,6):
        if os.path.exists('server.log.' + str(i)):
            os.remove('server.log.' + str(i))

# Создание временной папки, если она была удалена
if os.path.exists('temp') == False:
    os.makedirs('temp')

# Конфигурация логгера
dictConfig({
    'version' : 1,
    'formatters' : {
        'simple' : {
            'format' : '%(levelname)-8s | %(message)s'
        }
    },
    'handlers' : {
        'console' : {
            'class' : 'logging.StreamHandler',
            'level' : 'DEBUG',
            'formatter' : 'simple',
            'stream' : 'ext://sys.stdout'
        },
        'file' : {
            'class' : 'logging.handlers.RotatingFileHandler',
            'level' : 'DEBUG',
            'maxBytes' : 16 * 1024 * 1024,
            'backupCount' : 5,
            'formatter' : 'simple',
            'filename' : 'server.log'
        }
    },
    'loggers' : {
        'console' : {
            'level' : 'DEBUG',
            'handlers' : ['console'],
            'propagate' : 'no'
        },
        'file' : {
            'level' : 'DEBUG',
            'handlers' : ['file'],
            'propagate' : 'no'
        }
    },
    'root' : {
        'level' : 'DEBUG',
        'handlers' : ['console', 'file']
    }
})

app = Flask(__name__)
auth = HTTPBasicAuth()
max_content_length = 16 * 1024 * 1024
f_source_data = 'data/source_data.txt'
f_w2v_model = 'data/w2v_model.bin'
f_net_model = 'data/net_model.txt'
f_net_weights = 'data/net_final_weights.h5'


def limit_content_length():
    ''' Декоратор для ограничения размера передаваемых клиентом данных. '''
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.content_length > max_content_length:                
                log('превышен максимальный размер передаваемых данных ({:.2f} кБ)'.format(request.content_length/1024), request.remote_addr, 'error')
                return make_response(jsonify({'error': 'Maximum data transfer size exceeded, allowed only until {: .2f} kB.'.format(max_content_length/1024)}), 413)
            elif request.content_length == 0:
                log('тело запроса не содержит данных', request.remote_addr, 'error')
                return make_response(jsonify({'error': 'The request body contains no data.'}), 400)
            elif request.json == None:
                log('тело запроса содержит неподдерживаемый тип данных', request.remote_addr, 'error')
                return make_response(jsonify({'error': 'The request body contains an unsupported data type. Only json is supported'}), 415)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def log(message, addr=None, level='info'):
    ''' Запись сообщения в лог файл с уровнем INFO или ERROR. По умолчанию используется INFO.
    1. addr - строка с адресом подключённого клиента
    2. message - сообщение
    3. level - уровень логгирования, может иметь значение либо 'info', либо 'error' '''
    if level == 'info':
        if addr == None:
            app.logger.info(datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
        else:
            app.logger.info(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
    elif level == 'error':
        if addr == None:
            app.logger.error(datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
        else:
            app.logger.error(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)

sr = None
pr = None
http_server = None
# Получение графа вычислений tensorflow по умолчанию (для последующей передачи в другой поток)
graph = get_default_graph()


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'The requested URL was not found on the server.'}), 404)


@app.errorhandler(405)
def method_not_allowed(error):
    return make_response(jsonify({'error': 'The method is not allowed for the requested URL.'}), 405)


@app.errorhandler(500)
def internal_server_error(error):
    print(error)
    return make_response(jsonify({'error': 'The server encountered an internal error and was unable to complete your request.'}), 500)


@auth.get_password
def get_password(username):
    # login testbot, password test
    if username == 'testbot':
        return 'test'


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access.'}), 401)


@app.route('/', methods=['GET'])
@app.route('/chatbot', methods=['GET'])
def root():
    return redirect('/chatbot/about')


@app.route('/chatbot/about', methods=['GET'])
@auth.login_required
def about():
    ''' Возвращает информацию о проекте. '''
    return jsonify({'about':'о проекте что-то там, типо это крутой бот, изи внедрить и т.д.'})


@app.route('/chatbot/questions', methods=['GET'])
@auth.login_required
def questions():    
    ''' Возвращает список всех поддерживаемых ботом вопросов. '''
    p = Preparation()
    questions = p.get_questions(f_source_data)
    return jsonify({'questions':questions})


@app.route('/chatbot/speech-to-text', methods=['POST'])
@auth.login_required
@limit_content_length()
def speech_to_text():
    ''' Принимает .wav файл с записанной речью, распознаёт её с помощью PocketSphinx и возвращает распознанную строку. '''
    data = request.json
    data = data.get('wav')
    if data == None:
        log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    data = base64.b64decode(data)
    with open('temp/answer.wav', 'wb') as audio:
        audio.write(data)

    log('принят .wav размером {:.2f} кБ, сохранено в temp/answer.wav'.format(len(data)/1024), request.remote_addr)    
    answer = sr.stt('temp/answer.wav')
    log("распознано: '" + answer + "'", request.remote_addr)
    return jsonify({'text':answer})


@app.route('/chatbot/text-to-speech', methods=['POST'])
@auth.login_required
@limit_content_length()
def text_to_speech():
    ''' Принимает строку, синтезирует речь с помощью RHVoice и возвращает .wav файл с синтезированной речью. '''    
    data = request.json
    data = data.get('text')
    if data == None:
        log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    log("принято: '" + data + "'", request.remote_addr)
    tts(data, 'into_file', 'temp/answer.wav')

    with open('temp/answer.wav', 'rb') as audiofile:
        data = audiofile.read()
    
    log('создан .wav размером {:.2f} кБ, сохранено в temp/answer.wav'.format(len(data)/1024), request.remote_addr)
    data = base64.b64encode(data)
    return jsonify({'wav':data.decode()})


@app.route('/chatbot/text-to-text', methods=['POST'])
@auth.login_required
@limit_content_length()
def text_to_text():
    ''' Принимает строку с вопросом к боту и возвращает ответ в виде строки. '''
    data = request.json
    data = data.get('text')
    if data == None:
        log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    log("принято: '" + data + "'", request.remote_addr)
    with graph.as_default():
        answer = pr.predict(data)
    log("ответ: '" + answer + "'", request.remote_addr)
    return jsonify({'text':answer})

# Всего 5 запросов:
# 1. GET-запрос на /chatbot/about, вернёт инфу о проекте
# 2. GET-запрос на /chatbot/questions, вернёт список всех вопросов
# 3. POST-запрос на /chatbot/speech-to-text, принимает .wav-файл и возвращает распознанную строку
# 4. POST-запрос на /chatbot/text-to-speech, принимает строку и возвращает .wav-файл с синтезированной речью
# 5. POST-запрос на /chatbot/text-to-text, принимает строку и возвращает ответ бота в виде строки

# Что бы узнать свой локальный адрес в сети: sudo ifconfig | grep "inet addr"

def run(wsgi, host, port):
    ''' Автовыбор доступного порта (если указан порт 0), загрузка языковой модели и нейронной сети и запуск сервера.
    1. wsgi - True: запуск WSGI сервера, False: запуск тестового Flask сервера '''
    
    if port == 0: # Если был введён порт 0, то автовыбор любого доступного порта
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            port = sock.getsockname()[1]
            log('выбран порт ' + str(port))
            sock.close()
        except socket.gaierror:
            log('адрес ' + host + ':' + str(port) + ' некорректен', level='error')
            sock.close()
            return
        except OSError:
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')
            sock.close()
            return

    log('Flask v.' + flask_version + ', WSGIServer v.' + wsgi_version)    
    log('установлен максимальный размер принимаемых данных: {:.2f} кБ'.format(max_content_length/1024))

    log('инициализация языковой модели...')
    global sr
    sr = SpeechRecognition('from_file')
    
    log('загрузка модели seq2seq...')
    global pr
    print()
    pr = Prediction(f_net_model, f_net_weights, f_w2v_model)
    print()
    
    if wsgi:  
        log('WSGI сервер запущен на http://' + host + ':' + str(port))
        global http_server
        try:
            http_server = WSGIServer((host, port), app, log=app.logger, error_log=app.logger)
            http_server.serve_forever()
        except OSError:
            print()
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')
    else:
        log('запуск тестового Flask сервера...')
        try:
            app.run(host=host, port=port, threaded=True, debug=False)
        except OSError:
            print()
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')


def on_stop(*args):
    print()
    log('сервер остановлен')
    if http_server != None:
        http_server.close()
    sys.exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == "Linux":
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    host = '127.0.0.1'
    port = 5000
    
    # rest_server.py - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000
    # rest_server.py host:port - запуск WSGI сервера на host:port
    # rest_server.py -d - запуск тестового Flask сервера на 127.0.0.1:5000
    # rest_server.py -d host:port - запуск тестового Flask сервера на host:port    
    # rest_server.py -d localaddr:port - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом port
    # Что бы выбрать доступный порт автоматически, укажите в host:port или localaddr:port порт 0

    # run(False, host, port)

    if len(sys.argv) > 1:
        if sys.argv[1] == '-d': # запуск тестового Flask сервера в debug режиме
            if len(sys.argv) > 2:
                if sys.argv[2].find('localaddr') != -1 and sys.argv[2].find(':') != -1: # для определения адреса машины в локальной сети
                    command_line = "ifconfig | grep 'inet addr'"
                    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = proc.communicate()
                    out = out.decode()
                    err = err.decode()
                    host = out[out.find('inet addr:') + len('inet addr:'):]
                    host = host[:host.find(' ')]
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(False, host, port)
                elif sys.argv[2].count('.') == 3 and sys.argv[2].count(':') == 1: # host:port
                    host = sys.argv[2][:sys.argv[2].find(':')]
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(False, host, port)
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
            else:
                run(False, host, port)

        elif sys.argv[1].count('.') == 3 and sys.argv[1].count(':') == 1: # host:port
            host = sys.argv[1][:sys.argv[1].find(':')]
            port = int(sys.argv[1][sys.argv[1].find(':') + 1:])
            run(True, host, port)
        elif sys.argv[1] == 'help':
            print('\nПоддерживаемые варианты работы:')
            print('\tбез аргументов - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000')
            print('\thost:port - запуск WSGI сервера на host:port')
            print('\t-d - запуск тестового Flask сервера на 127.0.0.1:5000')
            print('\t-d host:port - запуск тестового Flask сервера на host:port')
            print('\t-d localaddr:port - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом port\n')
        else:
            print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
    else: # запуск WSGI сервера с автоопределением адреса машины в локальной сети
        command_line = "ifconfig | grep 'inet addr'"
        proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        out = out.decode()
        err = err.decode()
        host = out[out.find('inet addr:') + len('inet addr:'):]
        host = host[:host.find(' ')]
        port = 5000
        run(True, host, port)

'''
if sys.argv[1].count('*') > 0: # для задания максимальной длины принимаемых данных в виде выражения 
    print('это выражение!')
    temp = sys.argv[1]
    new_content_length = 1
    while temp.count('*') > 0:
        new_content_length *= int(temp[:temp.find('*')])
        temp = temp[temp.find('*') + 1:]
    new_content_length *= int(temp)
    max_content_length = new_content_length
    print(max_content_length)
elif sys.argv[1].isdigit() == True: # для задания максимальной длины принимаемых данных в виде одного числа 
    print('это просто число!')
    print(int(sys.argv[1]))
    max_content_length = int(sys.argv[1]) * 1024
    print(max_content_length)
'''