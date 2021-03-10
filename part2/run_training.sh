#!/bin/bash

python assign2.py 40 256 256 relu 1 glove > glove_relu_b1_e40_b256_h256.txt
python assign2.py 40 256 256 relu 1 fastText > fastText_relu_b1_e40_b256_h256.txt
python assign2.py 40 256 256 relu 1 word2vec > word2vec_relu_b1_e40_b256_h256.txt
python assign2.py 40 256 0 relu 1 glove > glove_relu_b1_e40_b256_h0.txt
python assign2.py 40 256 0 relu 1 fastText > fastText_relu_b1_e40_b256_h0.txt
python assign2.py 40 256 0 relu 1 word2vec > word2vec_relu_b1_e40_b256_h0.txt

python assign2.py 40 256 256 relu 0 glove > glove_relu_b0_e40_b256_h256.txt
python assign2.py 40 256 256 relu 0 fastText > fastText_relu_b0_e40_b256_h256.txt
python assign2.py 40 256 256 relu 0 word2vec > word2vec_relu_b0_e40_b256_h256.txt
python assign2.py 40 256 0 relu 0 glove > glove_relu_b0_e40_b256_h0.txt
python assign2.py 40 256 0 relu 0 fastText > fastText_relu_b0_e40_b256_h0.txt
python assign2.py 40 256 0 relu 0 word2vec > word2vec_relu_b0_e40_b256_h0.txt

python assign2.py 40 256 256 sigmoid 1 glove > glove_sigmoid_b1_e40_b256_h256.txt
python assign2.py 40 256 256 sigmoid 1 fastText > fastText_sigmoid_b1_e40_b256_h256.txt
python assign2.py 40 256 256 sigmoid 1 word2vec > word2vec_sigmoid_b1_e40_b256_h256.txt
python assign2.py 40 256 0 sigmoid 1 glove > glove_sigmoid_b1_e40_b256_h0.txt
python assign2.py 40 256 0 sigmoid 1 fastText > fastText_sigmoid_b1_e40_b256_h0.txt
python assign2.py 40 256 0 sigmoid 1 word2vec > word2vec_sigmoid_b1_e40_b256_h0.txt

python assign2.py 40 256 256 sigmoid 0 glove > glove_sigmoid_b0_e40_b256_h256.txt
python assign2.py 40 256 256 sigmoid 0 fastText > fastText_sigmoid_b0_e40_b256_h256.txt
python assign2.py 40 256 256 sigmoid 0 word2vec > word2vec_sigmoid_b0_e40_b256_h256.txt
python assign2.py 40 256 0 sigmoid 0 glove > glove_sigmoid_b0_e40_b256_h0.txt
python assign2.py 40 256 0 sigmoid 0 fastText > fastText_sigmoid_b0_e40_b256_h0.txt
python assign2.py 40 256 0 sigmoid 0 word2vec > word2vec_sigmoid_b0_e40_b256_h0.txt