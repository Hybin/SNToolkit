from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
from tqdm import tqdm
import numpy as np
import pickle
import utils


class Classifier(object):
    def __init__(self, config, processor, embedding_dim=200, bi_rnn_units=200, epochs=10):
        self.conf = config
        self.embedding_dim = embedding_dim
        self.bi_rnn_units = bi_rnn_units
        self.epochs = epochs
        self.processor = processor

    def build(self, train=True):
        if train:
            x, y = self.processor.vectorize()
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=24)

        with open(self.conf.vocab_tags.format(self.processor.task), "rb") as fp:
            vocabulary, word_label = pickle.load(fp)

        # Model Configuration
        model = Sequential()

        model.add(Embedding(len(vocabulary), self.embedding_dim, mask_zero=True))
        model.add(Bidirectional(LSTM(self.bi_rnn_units // 2, return_sequences=True)))

        crf = CRF(len(word_label), sparse_target=True)
        model.add(crf)

        model.summary()
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

        if train:
            return model, (train_x, train_y), (test_x, test_y)
        else:
            return model, (vocabulary, word_label)

    def train(self):
        model, (train_x, train_y), (test_x, test_y) = self.build(train=True)

        # Store the validate set
        with open(self.conf.validate_set.format(self.processor.task), "wb") as fp:
            pickle.dump((test_x, test_y), fp)

        # Train the model
        model.fit(train_x, train_y, batch_size=16, epochs=self.epochs, validation_data=[test_x, test_y])
        model.save(self.conf.model_path.format(self.processor.task))

    def test(self):
        model, (vocabulary, word_label) = self.build(train=False)

        # Extract the validate set
        with open(self.conf.validate_set.format(self.processor.task), "rb") as fp:
            test_x, test_y = pickle.load(fp)

        # Load the model
        print("Loading the model ** {} ** ...".format(self.conf.model_path.format(self.processor.task)))
        model.load_weights(self.conf.model_path.format(self.processor.task))

        print("Done!")
        print("Predicting the labels...")
        predictions = model.predict(test_x)
        pred_y, gold_y, count = [], [], 0

        for prediction in tqdm(predictions, desc="Predict"):
            sentence = test_x[count]
            len_of_sentence = utils.get_length(sentence)
            prediction = prediction[-len_of_sentence:]
            labels = [word_label[label[0]] for label in test_y[count][-len_of_sentence:]]
            gold_y.append(labels)

            label_index = [word_label[np.argmax(row)] for row in prediction]
            pred_y.append(label_index)

            count += 1

        report = classification_report(gold_y, pred_y)
        print(report)


