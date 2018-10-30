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
import base64
import json
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

# Конфигурация логгера
dictConfig({
    'version' : 1,
    'formatters' : {
        'simple' : {
            'format' : '%(levelname)s\t| %(message)s'
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
            'maxBytes' : 10 * 1024 * 1024,
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
max_content_length = 1 * 1 * 1024
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
                log(request.remote_addr, 'превышен максимальный размер передаваемых данных ({:.2f} кБ)'.format(request.content_length/1024), 'error')
                return make_response(jsonify({'error': 'Maximum data transfer size exceeded, allowed only until {: .2f} kB.'.format(max_content_length/1024)}), 413)
            elif request.content_length == 0:
                log(request.remote_addr, 'тело запроса не содержит данных', 'error')
                return make_response(jsonify({'error': 'The request body contains no data.'}), 400)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def log(addr, message, level='info'):
    ''' Запись сообщения в лог файл с уровнем INFO или ERROR. По умолчанию используется INFO.
    1. addr - строка с адресом подключённого клиента
    2. message - сообщение
    3. level - уровень логгирования, может иметь значение либо 'info', либо 'error' '''
    if level == 'info':
        app.logger.info(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
    elif level == 'error':
        app.logger.error(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)


log('localhost', 'Flask v.' + flask_version + ', WSGIServer v.' + wsgi_version)
log('localhost', 'запуск сервера...')
log('localhost', 'установлен максимальный размер принимаемых данных: {:.2f} кБ'.format(max_content_length/1024))
log('localhost', 'инициализация языковой модели...')
sr = SpeechRecognition('from_file')

log('localhost', 'загрузка модели seq2seq...')
pr = Prediction(f_net_model, f_net_weights, f_w2v_model)
# Получение графа вычислений tensorflow по умолчанию (для последующей передачи в другой поток)
graph = get_default_graph()


# Код 302 переопределить непонятно как
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
    if username == 'server':
        return 'python'


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
    data = base64.b64decode(data.get('wav'))
    with open('temp/answer.wav', 'wb') as audio:
        audio.write(data)

    log(request.remote_addr, 'принят .wav размером {:.2f} кБ, сохранено в temp/answer.wav'.format(len(data)/1024))    
    answer = sr.stt('temp/answer.wav')
    log(request.remote_addr, "распознано: '" + answer + "'")
    return jsonify({'text':answer})


@app.route('/chatbot/text-to-speech', methods=['POST'])
@auth.login_required
@limit_content_length()
def text_to_speech():
    ''' Принимает строку, синтезирует речь с помощью RHVoice и возвращает .wav файл с синтезированной речью. '''    
    data = request.json
    data = data.get('text')
    log(request.remote_addr, "принято: '" + data + "'")
    tts(data, 'into_file', 'temp/answer.wav')

    with open('temp/answer.wav', 'rb') as audiofile:
        data = audiofile.read()
    
    log(request.remote_addr, 'создан .wav размером {:.2f} кБ, сохранено в temp/answer.wav'.format(len(data)/1024))
    data = base64.b64encode(data)
    return jsonify({'wav':data.decode()})


@app.route('/chatbot/text-to-text', methods=['POST'])
@auth.login_required
@limit_content_length()
def text_to_text():
    ''' Принимает строку с вопросом к боту и возвращает ответ в виде строки. '''
    data = request.json
    data = data.get('text')
    log(request.remote_addr, "принято: '" + data + "'")
    with graph.as_default():
        answer = pr.predict(data)
    log(request.remote_addr, "ответ: '" + answer + "'")
    return jsonify({'text':answer})

# Всего 5 запросов:
# 1. GET-запрос на /chatbot/about, вернёт инфу о проекте
# 2. GET-запрос на /chatbot/questions, вернёт список всех вопросов
# 3. POST-запрос на /chatbot/speech-to-text, принимает .wav-файл и возвращает распознанную строку
# 4. POST-запрос на /chatbot/text-to-speech, принимает строку и возвращает .wav-файл с синтезированной речью
# 5. POST-запрос на /chatbot/text-to-text, принимает строку и возвращает ответ бота в виде строки

# Что бы узнать свой локальный адрес в сети: sudo ifconfig | grep "inet addr"


if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 5000), app, log=app.logger)
    http_server.serve_forever()

    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)