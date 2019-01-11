from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from gensim.models.keyedvectors import KeyedVectors
from keras import regularizers
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pickle
import gc
import keras
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
import scipy
from sklearn.preprocessing import MinMaxScaler


def train_word2vec(documents, embedding_dim):
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('../Pubmed_JET/words.txt', binary=False)
    print("JET is ready...")
    word_vector = model.wv
    return word_vector

def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except:
            embedding_matrix[i] = np.zeros(100)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix

def word_embed_meta_data(documents, embedding_dim):
    tokenizer = Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
    tokenizer.fit_on_texts(documents)
    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix

def create_train_dev_set(tokenizer, sentences_pair, is_similar, num_classes, max_sequence_length, validation_split_ratio):
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
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val

def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
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


def train_model(self, sentences_pair, simms, num_classes, embedding_meta_data, model_save_directory='./'):
    tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

    train_data_x1, train_data_x2, train_labels, leaks_train, \
    val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                           simms, num_classes,self.max_sequence_length,
                                                                           self.validation_split_ratio)

    if train_data_x1 is None:
        print("++++ !! Failure: Unable to train model ++++")
        return None

    nb_words = len(tokenizer.word_index) + 1

    # Creating word embedding layer
    embedding_layer = Embedding(nb_words, self.embedding_dim,
#                                    weights=[embedding_matrix],
                                input_length=self.max_sequence_length, trainable=True)

    # Creating LSTM Encoder
    lstm_layer = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))

    # Creating LSTM Encoder layer for First Sentence
    sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    # Creating LSTM Encoder layer for Second Sentence
    sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    x2 = lstm_layer(embedded_sequences_2)

    # Creating leaks input
    leaks_input = Input(shape=(3,))
    leaks_dense = Dense(int(self.number_dense_units/2), activation=self.activation_function,kernel_regularizer=regularizers.l2(0.01))(leaks_input)

    # Merging two LSTM encodes vectors from sentences to
    # pass it to dense layer applying dropout and batch normalisation
    merged = concatenate([x1, x2, leaks_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(self.rate_drop_dense)(merged)
    merged = Dense(self.number_dense_units, activation=self.activation_function,kernel_regularizer=regularizers.l2(0.01))(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(self.rate_drop_dense)(merged)
#        preds = Dense(num_classes, activation='sigmoid')(merged)
#        merged = Dense(self.number_dense_units, activation=self.activation_function,kernel_regularizer=regularizers.l2(0.0001))(merged)
#        merged = BatchNormalization()(merged)
#        merged = Dropout(self.rate_drop_dense)(merged)

    preds = Dense(output_dim=num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(merged)
#        model.summary()


    model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

    checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + STAMP + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

    his = model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
              validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
              epochs=500, batch_size=600, shuffle=True,verbose=1,
              callbacks=[early_stopping])

    pred1 = model.predict([val_data_x1,val_data_x2, leaks_val])
    pred1 = np.argmax(pred1,axis = 1)
    lables = np.argmax(val_labels,axis=1)
    score1 = scipy.stats.pearsonr(pred1, lables)[0]
    score=[]
    score.append(score1)


    pred2 = model.predict([train_data_x1, train_data_x2, leaks_train])
    pred2 = np.argmax(pred2,axis = 1)
    lables = np.argmax(train_labels,axis=1)
    score2 = scipy.stats.pearsonr(pred2, lables)[0]
    score.append(score2)
    return model, score,his

if __name__ == '__main__':
    df = pd.read_csv('../nlp-notebooks/dataset.csv')
    ori_sentences1 = list(df['sent_1'])
    ori_sentences2 = list(df['sent_2'])
    ori_is_similar = list(df['sim'])
    ori_sentences_pair = [(x1, x2) for x1, x2 in zip(ori_sentences1, ori_sentences2)]

    num_classes = 6

    scalar = MinMaxScaler(feature_range=(0, 5))
    temp = np.array(is_similar)
    scalar.fit(temp.reshape(700,1))
    temp = scalar.transform(temp.reshape(700,1))
    #round_up = np.rint(temp)
    round_up = np.floor(temp)
    round_up = keras.utils.to_categorical(round_up, num_classes)


    df = pd.read_csv('/Users/loretta/Documents/Project/nlp-notebooks/test.csv')
    df = df.drop(columns=['Unnamed: 0'])
    test_sentences1 = list(df['sent_1'])
    test_sentences2 = list(df['sent_2'])
    test_is_similar = list(df['sim'])
    test_sentences_pair = [(x1, x2) for x1, x2 in zip(test_sentences1, test_sentences2)]


    scalar = MinMaxScaler(feature_range=(0, 5))
    temp = np.array(test_is_similar)
    temp = scalar.transform(temp.reshape(50,1))
    test_round_up = np.floor(temp)
    test_round_up = keras.utils.to_categorical(test_round_up, num_classes)

    tokenizer, embedding_matrix = word_embed_meta_data(ori_sentences1 + ori_sentences2,  siamese_config['EMBEDDING_DIM'])

    embedding_meta_data = {
    	'tokenizer': tokenizer,
    	'embedding_matrix': embedding_matrix
    }


    model, score,his = train_model(sentences_pair, round_up, num_classes, embedding_meta_data, model_save_directory='./')
    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentences_pair,  siamese_config['MAX_SEQUENCE_LENGTH'])

    pred2 = model.predict([test_data_x1, test_data_x2, leaks_test])
    pred2 = np.argmax(pred2,axis=1)
    label = np.argmax(test_round_up,axis=1)
    #pred2 = [item[1] for item in pred2]
    #temp = [item[0] for item in temp]
    print(scipy.stats.pearsonr(pred2, label)[0])
    import matplotlib.pyplot as plt
    tmp = his.history
    tmp = pd.DataFrame(tmp)
    #tmp.to_csv('history_reg.csv')
    fig = plt.figure(1)
    plt.plot(tmp['acc'], 'r:')
    plt.plot(tmp['val_acc'], 'g-')
