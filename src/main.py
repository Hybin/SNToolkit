from config import Config
from processor import Processor
from classifier import Classifier
from annotator import Annotator

if __name__ == '__main__':
    # Load the config file
    print("Load the configuration file...")
    config = Config()

    print("done!")
    # Load the train data
    processor = Processor(config, "ner")

    # Load the classifier
    classifier = Classifier(config, processor)

    print("Training the model...")
    classifier.train()

    print("Begin to compute the F1-Score")
    classifier.test()
    # Load the annotator
    # annotator = Annotator(config, classifier)

    # results = annotator.write()

    print("complete!")
