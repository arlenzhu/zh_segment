# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: segment_model
Description : 
Author : arlen
date：18-7-12
------------------------------------------------- """
import numpy as np
from keras.layers import GRU, Embedding, Bidirectional, TimeDistributed, Dense
from keras_contrib.layers import CRF
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard


MAX_LEN = 399
VOCAB_SIZE = 5128
EMBEDDING_OUT_DIM = 64
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3
NUM_CLASS = 5


def bigru_crf_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(GRU(HIDDEN_UNITS // 2, return_sequences=True)))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf = CRF(NUM_CLASS, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def train():
    train_x = np.load('./training/train_x.np')
    train_y = np.load('./training/train_y.np')

    model = bigru_crf_model()

    checkpoint = ModelCheckpoint('./model_file/model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir='./model_file/tensorboard')

    callbackslist = [checkpoint, tensorboard]

    model.fit(train_x, train_y,
              batch_size=128,
              epochs=20,
              validation_split=0.2,
              callbacks=callbackslist)


if __name__ == "__main__":
    train()