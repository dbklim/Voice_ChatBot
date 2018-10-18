# Bot_seq2seq
Management system for virtual assistants of the retailer's sales manager using neural networks, early test version. 

Для обучения сети необходимо, что бы в data/ был файл source_data.txt, содержащий данные для обучения сети в формате
вопрос %% ответ. Что бы запустить обучение, достаточно выполнить train.sh. С указанными параметрами достигнута точность 98.62%
(1574 из 1596 правильных ответов).

Для запуска сети в режиме вопрос-ответ, достаточно выполнить run.sh. Вопросы следует задавать из source_data.txt.

Используется рекуррентная нейронная сеть, а если точнее - модель sequence2sequence. Слова переводятся в вектор с помощью 
word2vec из библиотеки gensim.

--------------------------------------------------------------------------------------------------------------------------

To train the network, you need to have data in data/source_data.txt that contains data for the network training in the 
format of a question %% answer. To start learning, just execute a train.sh. At the specified parameters the accuracy of 98.62% was achieved (1574 of 1596 correct answers).

To run the network in the question-answer mode, it suffices to execute run.sh. Questions should be asked from source_data.txt.

A recurrent neural network is used, or to be more precisely, the sequence2sequence model. Words are translated into a vector 
using word2vec from the gensim library.
