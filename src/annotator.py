from tqdm import tqdm
import numpy as np
import utils


class Annotator(object):
    def __init__(self, config, classifier):
        self.conf = config
        self.classifier = classifier

    # annotate the texts with label
    def write(self):
        model, (vocabulary, word_label) = self.classifier.build(train=False)
        model_path = self.conf.model_path.format(self.classifier.processor.task)

        # Load the model
        print("Loading the model ** {} ** ...".format(model_path))
        model.load_weights(model_path)

        print("Done")
        print("Predicting the labels...")

        sentences, test_x = self.classifier.processor.read_test_data()
        predictions = model.predict(test_x)

        results, count = [], 0
        for prediction in tqdm(predictions, desc="Predict"):
            sentence = test_x[count]
            len_of_sentence = len(sentences[count])
            prediction = prediction[-len_of_sentence:]
            labels = [word_label[np.argmax(row)] for row in prediction]

            result = []
            for i in range(len(labels)):
                result.append((sentences[count][i], labels[i]))

            results.append(result)

            count += 1

        # Write the data into the files
        with open(self.conf.result_path.format(self.classifier.processor.task), "w", encoding="utf-16") as fp:
            for result in tqdm(results, desc="Output the predictions"):
                if self.classifier.processor.task == "ner":
                    for word, label in result:
                        fp.write(word + " " + label + "\n")
                    fp.write("\n")
                else:
                    for word, label in result:
                        if label == "S" or label == "E":
                            fp.write(word + "  ")
                        else:
                            fp.write(word)

                    fp.write("\n")
