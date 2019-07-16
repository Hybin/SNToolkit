from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pickle
import utils
import re


class Processor(object):
    def __init__(self, conf, task):
        self.conf = conf
        self.task = task

    def load(self):
        """
        Load the training data
        :return: list of pairs based on sentences
        """
        sentences = []

        with open(self.conf.train_data.format(self.task), encoding="utf-16") as fp:
            lines, sentence = fp.readlines(), []

            for line in tqdm(lines, desc="Load the training data"):
                words = re.split(r'\s+', line.strip())

                if len(words) == 1 and words[0] == "":
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    sentence = []
                    continue
                else:
                    if self.task == "ner":
                        sentence.append(words)
                    else:
                        for word in words:
                            sentence += utils.annotate(word)

        fp.close()

        return sentences

    @staticmethod
    def _process_train_data(sentences, vocabulary, word_label, max_len=None, one_hot=False):
        """
        Transform the training sentence into vector
        :param sentences: list of pairs
        :param vocabulary: list
        :param word_label: list
        :param max_len: int
        :param one_hot: array
        :return: arrays
        """
        if max_len is None:
            max_len = max(len(sentence) for sentence in sentences)

        max_len += 1

        word_index = dict((word, index) for index, word in enumerate(vocabulary))
        x = [[word_index.get(word[0].lower(), 1) for word in sentence] for sentence in sentences]
        y = [[word_label.index(word[1]) for word in sentence] for sentence in sentences]

        # padding
        x = pad_sequences(x, max_len)
        y = pad_sequences(y, max_len, value=-1)

        if one_hot:
            y = np.eye(len(word_label), dtype='float32')[y]
        else:
            y = np.expand_dims(y, 2)

        return x, y

    def vectorize(self):
        sentences = self.load()

        word_count = Counter(word[0].lower() for sentence in sentences for word in sentence)
        vocabulary = [word for word, frequency in iter(word_count.items()) if frequency >= 2]
        word_label = self.conf.task_type[self.task]

        # Store the vocabulary and word_label
        with open(self.conf.vocab_tags.format(self.task), "wb") as fp:
            pickle.dump((vocabulary, word_label), fp)

        x, y = self._process_train_data(sentences, vocabulary, word_label)

        return x, y

    @staticmethod
    def _process_test_data(sentences, vocabulary, max_len=None):
        """
        Transform the test sentence to vector
        :param sentences: list
        :param vocabulary: list
        :param max_len: int
        :return: arrays
        """
        if max_len is None:
            max_len = max(len(sentence) for sentence in sentences)

        max_len += 1

        word_index = dict((word, index+1) for index, word in enumerate(vocabulary))
        x = [[word_index.get(word, 1) for word in sentence] for sentence in sentences]

        # padding
        x = pad_sequences(x, max_len)

        return x

    def read_test_data(self):
        # Load the test data - ner
        sentences = []

        with open(self.conf.test_data.format(self.task), encoding="utf-16") as fp:
            lines, sentence = fp.readlines(), []

            for line in tqdm(lines, desc="Loading the test data[{}]".format(self.task)):
                line = line.strip()

                if self.task == "ner":
                    if len(line) == 0 and len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    else:
                        sentence.append(line)
                else:
                    if len(line) == 0:
                        continue
                    else:
                        sentence = [word for word in line]
                        sentences.append(sentence)

        fp.close()

        # vectorize
        with open(self.conf.vocab_tags.format(self.task), "rb") as fp:
            vocabulary, word_label = pickle.load(fp)

        x = self._process_test_data(sentences, vocabulary)

        return sentences, x
