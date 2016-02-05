'''
The MIT License (MIT)

Copyright (c) 2016 Bonggun Shin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.linear_model import LogisticRegression


# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
import time

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM



class Timer(object):
    def __init__(self, name=None, logger=None):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.logger is None:
            if self.name:
                print '[%s]' % self.name,
            print 'Elapsed: %s' % (time.time() - self.tstart)

        else:
            if self.name:
                self.logger.info("[%s] Elapsed: %s" % (self.name, (time.time() - self.tstart)))
            else:
                self.logger.info('Elapsed: %s' % (time.time() - self.tstart))


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


class SentimentAnalyzer(object):
    def __init__(self):
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)



    def evaluate(self):
        classifier = LogisticRegression()
        classifier.fit(self.train_arrays, self.train_labels)
        score = classifier.score(self.test_arrays, self.test_labels)

        self.logger.info("Score is %s!!" % str(score))

    def evaluate_nn(self):
        nn_model = Sequential()
        nn_model.add(Embedding(self.dim, 200))
        nn_model.add(LSTM(200))  # try using a GRU instead, for fun
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(1))
        nn_model.add(Activation('sigmoid'))

        nn_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")

        print("Train...")
        nn_model.fit(self.train_arrays, self.train_labels, batch_size=32, nb_epoch=3,
                  validation_data=(self.test_arrays, self.test_labels), show_accuracy=True)
        score, acc = nn_model.evaluate(self.test_arrays, self.test_labels,
                                    batch_size=32,
                                    show_accuracy=True)
        print('Test score:', score)
        print('Test accuracy:', acc)

    def make_word_data(self):
        # self.train_word_data, self.test_word_data = self._get_data(self.train_sources, self.test_sources)
        train_sentences = LabeledLineSentence(self.train_sources)
        self.train_word_data = train_sentences.to_array()

        self.n_train_neg = sum(1 for d in self.train_word_data if "TRAIN_NEG" in d[1][0])
        self.n_train_pos = sum(1 for d in self.train_word_data if "TRAIN_POS" in d[1][0])
        self.n_train = self.n_train_neg+self.n_train_pos

        test_sentences = LabeledLineSentence(self.test_sources)
        self.test_word_data  = test_sentences.to_array()

        self.n_test_neg = sum(1 for d in self.test_word_data if "TEST_NEG" in d[1][0])
        self.n_test_pos = sum(1 for d in self.test_word_data if "TEST_POS" in d[1][0])
        self.n_test = self.n_test_neg+self.n_test_pos



class SentimentBoW(SentimentAnalyzer):
    def __init__(self, train_sources, test_sources, vocab_sources):
        super(SentimentBoW, self).__init__()
        self.logger.info("running %s" % ' '.join(sys.argv))
        self.train_sources = train_sources
        self.test_sources = test_sources
        self.vocab_sources = vocab_sources

        self.vocab = None
        self.train_word_data = None
        self.test_word_data = None



    def make_vocab(self):
        sentences = LabeledLineSentence(self.vocab_sources)
        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
        model.build_vocab(sentences.to_array())
        self.vocab = model.vocab
        self.vocab_len = len(self.vocab)

    def to_sparse_vector(self, words):
        sparse_vec = np.zeros([self.vocab_len,], dtype=np.int8)
        for w in words:
            try:
                sparse_vec[self.vocab[w].index] = 1
            except KeyError:
                pass
        return sparse_vec

    def make_dataset(self):
        if self.vocab is None:
            with Timer("make_vocab", self.logger):
                self.make_vocab()

        if self.train_word_data is None or self.test_word_data is None:
            with Timer("make_word_data", self.logger):
                self.make_word_data()

        with Timer("make dataset", self.logger):
            self.train_arrays = np.zeros((self.n_train, self.vocab_len))
            self.train_labels = np.zeros(self.n_train)

            for i in range(self.n_train):
                if "TRAIN_NEG" in self.train_word_data[i][1][0]:
                    self.train_arrays[i] = self.to_sparse_vector(self.train_word_data[i][0])
                    self.train_labels[i] = 0

                else: # "TRAIN_NEG"
                    self.train_arrays[i] = self.to_sparse_vector(self.train_word_data[i][0])
                    self.train_labels[i] = 1

                if i%(self.n_train/10)==0:
                    self.logger.info("making train %s done" % str(i/(self.n_train/100)+10))

            del self.train_word_data

            self.test_arrays = np.zeros((self.n_test, self.vocab_len))
            self.test_labels = np.zeros(self.n_test)

            for i in range(self.n_test):
                if "TEST_NEG" in self.test_word_data[i][1][0]:
                    self.test_arrays[i] = self.to_sparse_vector(self.test_word_data[i][0])
                    self.test_labels[i] = 0

                else: # "TEST_NEG"
                    self.test_arrays[i] = self.to_sparse_vector(self.test_word_data[i][0])
                    self.test_labels[i] = 1

                if i%(self.n_test/10)==0:
                    self.logger.info("making test %s done" % str(i/(self.n_test/100)+10))

            del self.test_word_data




class SentimentDoc2Vec(SentimentAnalyzer):
    def __init__(self, train_sources, test_sources, vocab_sources, dim=50):
        super(SentimentDoc2Vec, self).__init__()
        self.logger.info("running %s" % ' '.join(sys.argv))
        self.train_sources = train_sources
        self.test_sources = test_sources
        self.vocab_sources = vocab_sources
        self.dim = dim

        self.train_word_data = None
        self.test_word_data = None
        self.model = None

    def make_model(self, fname):
        if os.path.isfile(fname):
            with Timer("Load model from a file", self.logger):
                self.model = Doc2Vec.load('./imdb.d2v')
                self.dim = self.model.vector_size

        else:
            with Timer("build model from documents", self.logger):
                sentences = LabeledLineSentence(self.vocab_sources)
                model = Doc2Vec(min_count=1, window=10, size=self.dim, sample=1e-4, negative=5, workers=7)
                model.build_vocab(sentences.to_array())

                for epoch in range(50):
                    self.logger.info('Epoch %d' % epoch)
                    model.train(sentences.sentences_perm())

                model.save(fname)
                self.model = model

    def make_dataset(self):
        if self.train_word_data is None or self.test_word_data is None:
            with Timer("make_word_data", self.logger):
                self.make_word_data()

        if self.model is None:
            self.logger.info('model not ready')
            return

        with Timer("make dataset", self.logger):

            self.train_arrays = np.zeros((self.n_train, self.dim))
            self.train_labels = np.zeros(self.n_train)

            for i in range(self.n_train):
                if "TRAIN_NEG" in self.train_word_data[i][1][0]:
                    self.train_arrays[i] = self.model.infer_vector(self.train_word_data[i][0])
                    self.train_labels[i] = 0

                else: # "TRAIN_POS"
                    self.train_arrays[i] = self.model.infer_vector(self.train_word_data[i][0])
                    self.train_labels[i] = 1

                if i%(self.n_train/10)==0:
                    self.logger.info("making train %s done" % str(i/(self.n_train/100)+10))

            self.test_arrays = np.zeros((self.n_test, self.dim))
            self.test_labels = np.zeros(self.n_test)

            for i in range(self.n_test):
                if "TEST_NEG" in self.test_word_data[i][1][0]:
                    self.test_arrays[i] = self.model.infer_vector(self.test_word_data[i][0])
                    self.test_labels[i] = 0

                else: # "TEST_POS"
                    self.test_arrays[i] = self.model.infer_vector(self.test_word_data[i][0])
                    self.test_labels[i] = 1

                if i%(self.n_test/10)==0:
                    self.logger.info("making test %s done" % str(i/(self.n_test/100)+10))


train_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
test_sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}
vocab_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

# bow = SentimentBoW(train_sources, test_sources, vocab_sources)
# bow.make_dataset()
# bow.evaluate()

# d2v = SentimentDoc2Vec(train_sources, test_sources, vocab_sources, 100)
# d2v.make_model('./imdb.d2v')
# d2v.make_dataset()
# d2v.evaluate()
# d2v.evaluate_nn()


