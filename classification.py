import sys
import os
import numpy as np
import operator
import h5py
from gensim.models import word2vec
from segment import segmenter, del_stops
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Input, Dense, Dropout, Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
import argparse
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D, concatenate
from keras.models import Model
from IPython import embed


def get_embedding_dict(wordvec):
    """for glove"""
    embeddings_index = {}
    with open(wordvec)as f:
        for line in f:
            values = line.strip().split(" ")
            word = values[0].replace(u"\xa0", " ")
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def get_embedding_matrix(word_index, embedding_dict, num_words, embedding_dim):
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if embedding_dict.vocab.get(word):
            # if embedding_dict.get(word): """for dict from glove"""
            embedding_vector = embedding_dict[word]
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def read_data(src_dir, training):
    tags = []
    articles = []
    for dir_path, dir_names, file_names in os.walk(src_dir):
        for filename in file_names:
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                text = f.read()
            segs = segmenter(text)
            article = " ".join(del_stops(segs))
            article = del_stops(segs)
            if training:
                tag = dir_path.split("/")[-1]
                tags.append(tag)
            articles.append(article)
    if training:
        assert len(tags) == len(articles)
    tags_list = ["dry", "normal", "oily"]
    return (tags, articles, tags_list)


def f1_score(y_true, y_pred):
    y_pred /= K.max(y_pred, 1, keepdims=True)
    # thresh = 0.45
    # y_pred = K.cast(K.greater(y_pred, thresh), dtype='float32')
    y_pred = K.cast(K.equal(y_pred, 1.0), dtype='float32')
    tp = K.sum(y_true * y_pred, axis=0)

    precision = tp / (K.sum(y_pred, axis=0) + K.epsilon())
    recall = tp / (K.sum(y_true, axis=0) + K.epsilon())
    return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))


def f1score(y_true, y_pred, thresh=None):
    y_pred /= np.max(y_pred, 1).reshape(-1, 1)
    if thresh:
        y_pred = np.where(y_pred > thresh, 1, 0)
    else:
        y_pred = np.where(y_pred == 1.0, 1, 0)
    tp = np.sum(y_true * y_pred, 0)
    accuracy = tp.sum() / len(y_true)
    precision = tp / (np.sum(y_pred, 0) + 0.0000000001)
    recall = tp / (np.sum(y_true, 0) + 0.0000000001)
    f1 = 2 * ((precision * recall) / (precision + recall + 0.0000000001))
    return precision, recall, f1, accuracy


def embedding(X, tag):
    embedding_dim = 128
    model = word2vec.Word2Vec(X, size=embedding_dim)
    X = [" ".join(i) for i in X]

    tk = Tokenizer()
    tk.fit_on_texts(X)
    seq = tk.texts_to_sequences(X)
    # pre_pad_seq=pad_sequences(seq, maxlen=None,padding="pre" ,dtype='int32')
    post_pad_seq = pad_sequences(
        seq, maxlen=None, padding="post", dtype='int32')
    pad_seq = np.fliplr(post_pad_seq)
    # pad_seq = np.hstack((pre_pad_seq,pad_seq))

    # embedding_index=get_embedding_dict("glove.840B.300d.txt")
    embedding_index = model.wv
    num_words = len(tk.word_index) + 1
    embedding_matrix = get_embedding_matrix(
        tk.word_index, embedding_index, num_words, embedding_dim)
    return embedding_matrix, pad_seq


def split(seq, tag):
    randperm = np.random.permutation(len(tag))
    cutpoint = len(tag) // 4

    trainseq = seq[randperm[cutpoint:]]
    testseq = seq[randperm[:cutpoint]]
    train_tag = tag[randperm[cutpoint:]]
    test_tag = tag[randperm[:cutpoint]]

    return trainseq, testseq, train_tag, test_tag


class RNN(object):

    def main(self, X, tag):
        embedding_matrix, seq = embedding(X, tag)
        trainseq, testseq, train_tag, test_tag = split(seq, tag)
        model = Sequential()
        # model.add( embedding_index.get_keras_embedding()) """haven't figure
        # out how to use it"""
        model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[
                  embedding_matrix], input_length=seq.shape[1], trainable=True))
        model.add(LSTM(256, activation='tanh', dropout=0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.1))
        # model.add(Dense(256,activation="relu"))
        # model.add(Dropout(0.1))
        model.add(Dense(len(tag_list), activation='softmax'))
        # model.add(Dense(len(tag_list),activation='sigmoid'))
        return model, trainseq, testseq, train_tag, test_tag


class CNN(object):

    def main(self, X, tag):
        embedding_matrix, seq = embedding(X, tag)
        trainseq, testseq, train_tag, test_tag = split(seq, tag)
        main_input = Input(shape=(seq.shape[1],),
                           dtype='int32', name='main_input')
        submodel = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[
                             embedding_matrix], input_length=seq.shape[1], trainable=True)(main_input)
        submodels = []
        for kw in (3, 4, 5):    # kernel sizes
            conv = Conv1D(16, kw, padding='valid',
                          activation='relu', strides=2)(submodel)
            mp = GlobalMaxPooling1D()(conv)
            submodels.append(mp)

        big_model = concatenate(submodels)
        big_model = Dense(256, activation='relu')(big_model)
        big_model = Dropout(0.4)(big_model)
        big_model = Dense(len(tag_list), activation='softmax')(big_model)
        model = Model(inputs=main_input, outputs=big_model)

        return model, trainseq, testseq, train_tag, test_tag


class BOW(object):

    def main(self, X, tag):
        X = [" ".join(i) for i in X]
        vectorizer = TfidfVectorizer()
        seq = vectorizer.fit_transform(X).toarray()
        trainseq, testseq, train_tag, test_tag = split(seq, tag)
        nbwords = seq.shape[1]
        model = Sequential()
        model.add(Dense(1024, input_shape=(nbwords,)))
        model.add(Dropout(0.5))
        # model2.add(Dense(512,activation="relu"))
        model.add(Dense(len(tag_list), activation='softmax'))

        return model, trainseq, testseq, train_tag, test_tag


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--srcpath',
                        help='source directory path', type=str)
    parser.add_argument(
        '-m', '--mode', help='1:bow\n2:cnn\n3:rnn', type=int, default=1)
    parser.add_argument(
        '-s', '--split', help='train/test split', type=int, default=4)
    args = parser.parse_args()

    if args.srcpath:
        path = args.srcpath
    else:
        parser.print_help()
        sys.exit(1)

    Y, X, tag_list = read_data(path, 1)

    t = Tokenizer(split=" ", lower=False, filters="")
    t.fit_on_texts(Y)
    tag_list, _ = zip(*sorted(t.word_index.items(),
                              key=operator.itemgetter(1)))
    tag_list = list(tag_list)
    tag = t.texts_to_matrix(Y)[:, 1:]

    M = {1: BOW, 2: CNN, 3: RNN}[args.mode]()

    model, trainseq, testseq, train_tag, test_tag = M.main(X, tag)

    # model.compile(loss="binary_crossentropy",optimizer="adam",metrics=[f1_score])
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=[f1_score])
    earlystopping = EarlyStopping(
        monitor='val_f1_score', patience=3, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
    model.summary()
    model.save('best_model.h5')

    model.fit(trainseq, y=train_tag, batch_size=100, epochs=40, validation_data=(
        testseq, test_tag), shuffle=True, callbacks=[earlystopping, checkpoint])

    model.load_weights("best.hdf5")
    predict = model.predict(testseq)
    precision, recall, f1, accuracy = f1score(test_tag, predict)

    embed()

    # # setting thresholg for mulitilabel
    # maxth = 0
    # maxv = 0
    # for th in range(30, 60, 5):
    #     if f1score(train_tag, Y, th / 100) > maxv:
    #         maxv = (f1score(train_tag, Y, th / 100))
    #         maxth = th / 100
    #         print(maxv, maxth)
