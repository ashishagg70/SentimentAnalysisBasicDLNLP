Keshav Agarwal (203050039)
Ashish Aggarwal (203050015)
Debabrata Biswal (203050024)

Command to execute:
python3 assign2.py <epochs> <batch_size> <hidden_layer_size> <activation> <balanced/unbalalnced> <word_embedding>

epochs: number of epochs for training
batch_size: batch_size per step
hidden_layer_size: 0 for no hidden layer or number of neurons in the one hidden layer
activation: sigmoid or relu
balanced/unbalanced: 1 for balanced, 0 for balanced
word_embedding: glove for GloVe embedding
		word2vec for word2vec embedding
		fastText for fastText embedding

example:
python assign2.py 40 256 256 relu 1 glove


Note: For GUI follow readme.txt in SentimentAnalysis directory.