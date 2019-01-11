#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:45:25 2018

@author: loretta
"""

#import time,math,operator
from mpi4py import MPI
import numpy as np
import pandas as pd
import time
#import pickle
#import sklearn
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import scipy
from sklearn.preprocessing import MinMaxScaler

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start = time.time()

#print("now I have "+str(size)+"cores")
#=======================
#=load original dataset=
#=======================
df = pd.read_csv(
        'dataset.csv'
#        '/Users/loretta/lstm-siamese-text-similarity/dataset.csv'
        )
sentences1 = list(df['sent_1'])
sentences2 = list(df['sent_2'])
is_similar = list(df['sim'])
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]

#======================
#====pre-process=======
#======================
sent_pairs = []
with open('no_templates.txt', "r") as f:
    for line in f:
        ts = line.strip().split("\t")
        sent_pairs.append((ts[0], ts[1]))
df_noT = pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2"])

sentences1_noT = list(df_noT['sent_1'])
sentences2_noT= list(df_noT['sent_2'])
sentences_pair_noT = [(x1, x2) for x1, x2 in zip(sentences1_noT, sentences2_noT)]

#r1 = pd.read_csv('/Users/loretta/Documents/Project/nlp-notebooks/cui2vec_MetaMap.csv',header=None)

#s1 = [s[0] for s in sentences_pair]
#scale up 20 classes
scalar = MinMaxScaler(feature_range=(0, 20))
temp = np.array(is_similar)
scalar.fit(temp.reshape(750,1))
temp = scalar.transform(temp.reshape(750,1))
temp = np.rint(temp)
temp = np.hstack(temp)

comm.Barrier()
left_seq_len = 32
#data = r1[[1,2,3,4]]
#features = comm.bcast(data, root=0);
sentences = comm.bcast(sentences_pair,root=0)
sentences_noT = comm.bcast(sentences_pair_noT,root=0)
Y = comm.bcast(temp,root=0)
validation_split_ratio = 0.2
score=[]

comm.Barrier()

def create_test_data_ABCNN(tokenizer, test_sentences_pair, max_sequence_length):
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)

    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2

def create_test_data_LSTM(tokenizer, test_sentences_pair, max_sequence_length):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    test_sentences1 = [x[0] for x in test_sentences_pair]
    test_sentences2 = [x[1] for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]

    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2, leaks_test

def create_train_dev_set_ABCNN(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
#    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, val_data_1, val_data_2, labels_val

def create_train_dev_set_LSTM(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    sentences1 = [x[0] for x in sentences_pair]
    sentences2 = [x[1] for x in sentences_pair]
    train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]

    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

if rank == 0:
    sentences1 = [sent[0] for sent in sentences]
    sentences2 = [sent[1] for sent in sentences]
    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    tokenizer.fit_on_texts(sentences1+sentences2)
    model = load_model(
            'ABCNN.h5'
#            '/Users/loretta/Documents/Project/nlp-notebooks/ABCNN_no_pre-trained/1538462412.h5'
            )
    #new dataset provided, model needs to be updated...
    num_classes = 21


    round_up = keras.utils.to_categorical(Y, num_classes)

    train_data_x1, train_data_x2, train_labels, \
        val_data_x1, val_data_x2, val_labels  = create_train_dev_set_ABCNN(tokenizer, sentences, round_up, left_seq_len, validation_split_ratio)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=100, batch_size=64, shuffle=True,verbose=1,
                  callbacks =[early_stopping])

    pred1 = model.predict([val_data_x1,val_data_x2])
    pred1 = np.argmax(pred1,axis = 1)
    lables = np.argmax(val_labels,axis=1)
    score1 = scipy.stats.pearsonr(pred1, lables)[0]
    score.append(score1)

    pred2 = model.predict([train_data_x1, train_data_x2])
    pred2 = np.argmax(pred2,axis = 1)
    lables = np.argmax(train_labels,axis=1)
    score2 = scipy.stats.pearsonr(pred2, lables)[0]
    score.append(score2)

    test_data_x1, test_data_x2 = create_test_data_ABCNN(tokenizer, sentences, left_seq_len )
    pred = model.predict([test_data_x1, test_data_x2])
    pred = np.argmax(pred,axis = 1)
    lables = np.argmax(round_up,axis=1)
    score3= scipy.stats.pearsonr(pred, lables)[0]
    score.append(score3)
    score = pd.DataFrame(score,columns=['ABCNN'])
    pred = pd.DataFrame(pred,columns=['ABCNN'])
#    only do prediction...
#    test_data_x1, test_data_x2 = create_test_data_ABCNN(tokenizer, sentences, left_seq_len )
#    pred = model.predict([test_data_x1,test_data_x2])
#    pred = np.argmax(pred,axis=1)

if rank == 0:
#    for ABCNN, with clean dataset
    sentences1 = [sent[0] for sent in sentences_noT]
    sentences2 = [sent[1] for sent in sentences_noT]
    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    tokenizer.fit_on_texts(sentences1+sentences2)
    model = load_model(
            'ABCNN_noT.h5'
#            '/Users/loretta/Documents/Project/nlp-notebooks/ABCNN_no_pre-trained/1538462741.h5'
            )
    #new dataset provided, model needs to be updated...
    num_classes = 21
#
#    scalar = MinMaxScaler(feature_range=(0, 20))
#    temp = np.array(Y)
#    scalar.fit(temp.reshape(750,1))
#    temp = scalar.transform(temp.reshape(750,1))
#
#    round_up = np.rint(temp)
    round_up = keras.utils.to_categorical(Y, num_classes)

    train_data_x1, train_data_x2, train_labels, \
        val_data_x1, val_data_x2, val_labels  = create_train_dev_set_ABCNN(tokenizer, sentences_noT, round_up, left_seq_len, validation_split_ratio)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit([train_data_x1, train_data_x2], train_labels,
                  validation_data=([val_data_x1, val_data_x2], val_labels),
                  epochs=100, batch_size=64, shuffle=True,verbose=1,
                  callbacks =[early_stopping])

    pred1 = model.predict([val_data_x1,val_data_x2])
    pred1 = np.argmax(pred1,axis = 1)
    lables = np.argmax(val_labels,axis=1)
    score1 = scipy.stats.pearsonr(pred1, lables)[0]
    score.append(score1)

    pred2 = model.predict([train_data_x1, train_data_x2])
    pred2 = np.argmax(pred2,axis = 1)
    lables = np.argmax(train_labels,axis=1)
    score2 = scipy.stats.pearsonr(pred2, lables)[0]
    score.append(score2)

    test_data_x1, test_data_x2 = create_test_data_ABCNN(tokenizer, sentences_noT, left_seq_len )
    pred = model.predict([test_data_x1, test_data_x2])
    pred = np.argmax(pred,axis = 1)
    lables = np.argmax(round_up,axis=1)
    score3= scipy.stats.pearsonr(pred, lables)[0]
    score.append(score3)

    score = pd.DataFrame(score,columns=['ABCNN_noT'])
    pred = pd.DataFrame(pred,columns=['ABCNN_noT'])

#    only do prediction...
#    test_data_x1, test_data_x2 = create_test_data_ABCNN(tokenizer, sentences, left_seq_len )
#    pred = model.predict([test_data_x1,test_data_x2])
#    pred = np.argmax(pred,axis=1)

if rank == 0:
    sentences1 = [sent[0] for sent in sentences]
    sentences2 = [sent[1] for sent in sentences]

    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    tokenizer.fit_on_texts(sentences1+sentences2)
    model = load_model(
            'LSTM.h5'
#            '/Users/loretta/lstm-siamese-text-similarity/checkpoints/1538730928/lstm_50_50_0.17_0.20.h5'
            )

    num_classes = 21
    round_up = keras.utils.to_categorical(Y, num_classes)

    train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set_LSTM(tokenizer, sentences,
                                                                               round_up,left_seq_len,
                                                                               validation_split_ratio)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=64, shuffle=True,
                  callbacks=[early_stopping])

    pred1 = model.predict([val_data_x1, val_data_x2, leaks_val])
    pred1 = np.argmax(pred1,axis = 1)
    lables = np.argmax(val_labels,axis=1)
    score1 = scipy.stats.pearsonr(pred1, lables)[0]
    score.append(score1)

    pred2 = model.predict([train_data_x1, train_data_x2, leaks_train])
    pred2 = np.argmax(pred2,axis = 1)
    lables = np.argmax(train_labels,axis=1)
    score2 = scipy.stats.pearsonr(pred2, lables)[0]
    score.append(score2)

    test_data_x1, test_data_x2, leaks_test = create_test_data_LSTM(tokenizer, sentences, left_seq_len )
    pred = model.predict([test_data_x1, test_data_x2, leaks_test])
    pred = np.argmax(pred,axis = 1)
    lables = np.argmax(round_up,axis=1)
    score3= scipy.stats.pearsonr(pred, lables)[0]
    score.append(score3)
    score = pd.DataFrame(score,columns=['LSTM'])
    pred = pd.DataFrame(pred,columns=['LSTM'])
#    only do prediction...
#    test_data_x1, test_data_x2, leaks_test = create_test_data_LSTM(tokenizer, sentences, left_seq_len  )
#    pred = model.predict([test_data_x1,test_data_x2,leaks_test])
#    pred = np.argmax(pred,axis=1)
#
if rank == 0:
    sentences1 = [sent[0] for sent in sentences_noT]
    sentences2 = [sent[1] for sent in sentences_noT]

    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    tokenizer.fit_on_texts(sentences1+sentences2)
#    model = load_model('/Users/loretta/lstm-siamese-text-similarity/checkpoints/1538731505/lstm_50_50_0.17_0.20.h5')
    model = load_model('LSTM_noT.h5')
    num_classes = 21
    round_up = keras.utils.to_categorical(Y, num_classes)

    train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set_LSTM(tokenizer, sentences_noT,
                                                                               round_up,left_seq_len,
                                                                               validation_split_ratio)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=64, shuffle=True,
                  callbacks=[early_stopping])

    pred1 = model.predict([val_data_x1, val_data_x2, leaks_val])
    pred1 = np.argmax(pred1,axis = 1)
    lables = np.argmax(val_labels,axis=1)
    score1 = scipy.stats.pearsonr(pred1, lables)[0]
    score.append(score1)

    pred2 = model.predict([train_data_x1, train_data_x2, leaks_train])
    pred2 = np.argmax(pred2,axis = 1)
    lables = np.argmax(train_labels,axis=1)
    score2 = scipy.stats.pearsonr(pred2, lables)[0]
    score.append(score2)

    test_data_x1, test_data_x2, leaks_test = create_test_data_LSTM(tokenizer, sentences_noT, left_seq_len )
    pred = model.predict([test_data_x1, test_data_x2, leaks_test])
    pred = np.argmax(pred,axis = 1)
    lables = np.argmax(round_up,axis=1)
    score3= scipy.stats.pearsonr(pred, lables)[0]
    score.append(score3)
    score = pd.DataFrame(score,columns=['LSTM_noT'])
    pred = pd.DataFrame(pred,columns=['LSTM_noT'])

#    only do prediction...
#    test_data_x1, test_data_x2, leaks_test = create_test_data_LSTM(tokenizer, sentences, left_seq_len  )
#    pred = model.predict([test_data_x1,test_data_x2,leaks_test])
#    pred = np.argmax(pred,axis=1)

#if rank == 4:
#    model = load_model('/Users/loretta/Documents/Project/nlp-notebooks/NN.model')
#    pred = model.predict(data)
#    pred = np.argmax(pred,axis=1)
#
#if rank == 3:



#final = pd.DataFrame()
Prediction = comm.gather(pred,root=0)
Score = comm.gather(score, root=0)



if rank ==0:
#    finalPrediction = np.transpose(finalPrediction)
#    finalPrediction = pd.DataFrame(finalPrediction)
    finalPrediction = Prediction[0]
    for item in Prediction[1:]:
        finalPrediction = finalPrediction.merge(item,left_index=True,right_index=True)
    print("final result has gathered")
    print(finalPrediction.info)

#    for value in finalPrediction:
#        print(scipy.stats.pearsonr(finalPrediction[value], is_similar)[0])
    finalScore = Score[0]
    for item in Score[1:]:
        finalScore = finalScore.merge(item,left_index=True,right_index=True)
    s = []
    x_train, x_test, y_train, y_test = train_test_split(finalPrediction, Y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()

    clf = clf.fit(x_train, y_train)
    pred1=clf.predict(x_train)
    s.append(scipy.stats.pearsonr(pred1, y_train)[0])

    pred2=clf.predict(x_test)
    s.append(scipy.stats.pearsonr(pred2, y_test)[0])

    pred3=clf.predict(finalPrediction)
    s.append(scipy.stats.pearsonr(pred3, Y)[0])
    finalScore['final'] = s
    print(finalScore)
    print(time.time() - start)
#    num_classes = 21

#    scalar = MinMaxScaler(feature_range=(0, 20))
#    temp = np.array(is_similar)
#    scalar.fit(temp.reshape(750,1))
#    temp = scalar.transform(temp.reshape(750,1))
#    round_up = np.rint(temp)
#    round_up = [int(value) for value in round_up]
#
#    x_train, x_test, y_train, y_test = train_test_split(pred, round_up, test_size=0.33, random_state=42)
#
#    clf = DecisionTreeClassifier()
#
#    clf = clf.fit(x_train, y_train)
#    pred1=clf.predict(x_train)
#    s1.append(scipy.stats.pearsonr(pred1, y_train)[0])
#
#
#    pred2=clf.predict(x_test)
#    s2.append(scipy.stats.pearsonr(pred2, y_test)[0])
#
#    pred3=clf.predict(features)
#    s3.append(scipy.stats.pearsonr(pred3, rpund_up)[0])
#
