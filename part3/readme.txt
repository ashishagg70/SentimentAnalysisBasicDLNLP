Keshav Agarwal (203050039)
Ashish Aggarwal (203050015)
Debabrata Biswal (203050024)

Download the word2vec embedding for not pretrained model:
https://drive.google.com/file/d/1iiAFxYaUHpUXcd3468SHUM2fZV8kuxdF/view?usp=sharing

For pretraining download the fastText embedding and untar it:
https://drive.google.com/file/d/1iiAFxYaUHpUXcd3468SHUM2fZV8kuxdF/view?usp=sharing

Command to execute:
python3 assign3.py <epochs> <batch_size> <number_of_dense_layers> <pretrained/ trained> <rnnType> <number_of_rnn_layers>

epochs: number of epochs for training
batch_size: batch_size per step
number_of_dense_layers: number of dense layers
rnnType: rnn, lstm, bilstm, gru, bigru
number_of_rnn_layers: number of rnn layers

example:
To use the pretrained embedding:
python assign3.py 40 256 0 pretrained rnn 1

To use the trained embedding:
python assign3.py 40 256 0 trained rnn 1


Note: For GUI follow readme.txt in SentimentAnalysis directory.