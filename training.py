# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для обучения нейронной сети seq2seq. 
'''

import sys
import os
import numpy as np
from keras import __version__ as keras_version
import seq2seq
from seq2seq.models import SimpleSeq2Seq


class Training:
    ''' Предназначен для обучения нейронной сети seq2seq. '''
    def train(self, filename_in, filename_model, depth_model, number_training_cycles, number_epochs):
        ''' Запуск обучения модели seq2seq. Архитектура обученной сети и её веса сохраняются в net.h5 и net_weights.h5
        1. filename_in - имя .npz файла с векторным представлением слов в парах [вопрос,ответ]
        2. filename_model - имя .txt файла для сохранения параметров модели SimpleSeq2Seq
        3. depth - глубина модели seq2seq, задаёт число входных и выходных LSTM-слоёв
        4. number_training_cycles - количество циклов обучения сети
        5. number_epochs - количество эпох в одном цикле обучения сети '''

        print('[i] Загрузка данных из %s' % filename_in)
        npzfile = np.load(filename_in)
        Q, A = npzfile["Q"], npzfile["A"]

        Q = (Q + 1.0) * 0.5
        A = (A + 1.0) * 0.5

        num_ex, sequence_length, vec_size = Q.shape

        print('\tколичество примеров: %i' % num_ex)
        print('\tдлинна последовательности: %i' % sequence_length)
        print('\tразмер входа: %i' % vec_size)

        print('[i] Построение сети...')

        model = SimpleSeq2Seq(input_dim = vec_size, hidden_dim = vec_size, output_length = sequence_length, output_dim = vec_size, depth = depth_model)
        model.compile(loss = 'mse', optimizer = 'rmsprop')       
        self.__save_simpleseq2seq_model(filename_model, vec_size, vec_size, sequence_length, vec_size, depth_model, 'mse', 'rmsprop')

        print('[i] Обучение сети...')
        
        path_for_out = filename_in[:filename_in.find('/') + 1]

        for i in range(1, number_training_cycles + 1):
            model.fit(Q, A, epochs = number_epochs)
            print('\n\tсохранение промежуточного результата %i/%i' % (i, number_training_cycles))
            model.save_weights('%snet_%i_weights.h5' % (path_for_out, i))
            if i % 3 == 0 and i != number_training_cycles:
                self.__delete_tempfiles(i, path_for_out)
        print('[i] Обучение завершено')

        answer = model.predict(Q[0][np.newaxis,:])
        answer = answer * 2.0 - 1.0
          


    def __save_simpleseq2seq_model(self, filename, input_dim, hidden_dim, output_length, output_dim, depth, loss, optimizer):
        ''' Сохранение параметров модели SimpleSeq2Seq и параметров компиляции (optimizer и loss) в .txt файл. '''
        file_w = open(filename, 'w')
        file_w.write('input_dim=%i\n' % input_dim)
        file_w.write('hidden_dim=%i\n' % hidden_dim)
        file_w.write('output_length=%i\n' % output_length)
        file_w.write('output_dim=%i\n' % output_dim)
        file_w.write('depth=%i\n' % depth)
        file_w.write('loss=%s\n' % loss)
        file_w.write('optimizer=%s\n' % optimizer)
        file_w.close()
        # Сохранение через model.save() не корректно работает в данном случае, вероятно из-за использования в SimpleSeq2Seq моделей 
        # recurrentshop. При последующей загрузке модели model = load_model() возникает ошибка, решением которой является, по сути, 
        # построение сети заново со всеми параметрами. Для этого и был написан данный костыль.

    def __delete_tempfiles(self, i, path_for_out):
        ''' Удаление файлов с промежуточным результатом обучения сети (значения весов). '''
        for j in range(0, 3):
            path_weights = os.path.join(os.path.abspath(os.path.dirname(__file__)), '%snet_%i_weights.h5' % (path_for_out, (i - j)))
            os.remove(path_weights)




def main():
    t = Training()
    t.train('data/encoded_data.npz', 'data/net_model.txt', 2, 50, 5)


if __name__ == '__main__':
    print('Keras: %s' % keras_version)
    main()