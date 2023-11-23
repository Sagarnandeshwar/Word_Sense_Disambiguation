from sklearn.svm import SVC
import loader
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json

from sklearn.model_selection import GridSearchCV

from nltk.wsd import lesk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

STOPWORDS = stopwords.words('english')
ps = PorterStemmer()
wl = WordNetLemmatizer()

INSTANCE_FILE = '../dataset/multilingual-all-words.en.xml'
KEY_FILE = '../dataset/wordnet.en.key'
BOOTSTRAP = '../dataset/data.json'
SAMPLE_KEY = "d002.s002.t005"


class wsd_SVM():
    def __init__(self, train_dataset, test_dataset, train_label, test_label):
        self.models = {}
        self.label_encoders = {}
        self.count_vector = {}
        self.data = {}
        self.preproces(train_dataset, train_label)

    def preproces(self, dataset, label):
        for key in dataset.keys():
            lemma = dataset[key].lemma.decode()
            context = b" ".join(dataset[key].context).decode()
            correct_label = wordnet.synset_from_sense_key(label[key][0])
            if lemma not in self.data.keys():
                self.data[lemma] = {"context": [context], "sense": [correct_label]}
            else:
                self.data[lemma]["context"].append(context)
                self.data[lemma]["sense"].append(correct_label)

    def build_model(self, train_dataset, test_dataset, train_label, test_label):
        for key in self.data.keys():
            self.models[key] = SVC()
            self.count_vector[key] = CountVectorizer()
            self.label_encoders[key] = LabelEncoder()

        lemma_sense = {}
        for key in train_dataset:
            lemma = train_dataset[key].lemma.decode()
            correct_label = wordnet.synset_from_sense_key(train_label[key][0])
            if lemma not in lemma_sense.keys():
                lemma_sense[lemma] = [correct_label]
                continue
            lemma_sense[lemma].append(correct_label)

        for key in test_dataset:
            lemma = test_dataset[key].lemma.decode()
            correct_label = wordnet.synset_from_sense_key(test_label[key][0])
            if lemma not in lemma_sense.keys():
                lemma_sense[lemma] = [correct_label]
                continue
            lemma_sense[lemma].append(correct_label)

        for key in lemma_sense.keys():
            if key not in self.models.keys():
                continue
            self.label_encoders[key].fit(lemma_sense[key])

    def process_text(self, text):
        new_data = []
        for sentence in text:
            new_sentence = []
            sentence = word_tokenize(sentence)
            for words in sentence:
                w = words
                if w in STOPWORDS:
                    continue
                if not w.isalpha():
                    continue
                w = w.lower()
                # w = ps.stem(w)
                w = wl.lemmatize(w)
                new_sentence.append(w)
            new_data.append(" ".join(new_sentence))
        return new_data

    def train(self):
        for lemma in self.data.keys():
            lemma_context_org = self.data[lemma]["context"]
            lemma_sense = self.data[lemma]["sense"]

            lemma_context = self.process_text(lemma_context_org)

            self.count_vector[lemma].fit(lemma_context)
            self.label_encoders[lemma].fit(lemma_sense)

            X = self.count_vector[lemma].transform(lemma_context)
            Y = self.label_encoders[lemma].transform(lemma_sense)

            if len(np.unique(Y)) <= 1:
                synsets = wordnet.synsets(lemma)
                lesk_sense = lesk(lemma_context_org[0], lemma, synsets=synsets)
                self.models[lemma] = lesk_sense
                # self.models[lemma] = lemma_sense[0]
            else:
                self.models[lemma] = SVC(C=0.1, gamma=1, kernel='linear')
                self.models[lemma].fit(X, Y)
                """
                param_model = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 0.001],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                }
                model_grid = GridSearchCV(estimator=self.models[lemma], param_grid=param_model, verbose=1, cv=2, n_jobs=-1)
                model_grid.fit(X, Y)
                print(model_grid.best_estimator_)
                """

    def predict(self, key, test_dataset):
        lemma = test_dataset[key].lemma.decode()
        context = b" ".join(test_dataset[key].context).decode()

        if lemma not in self.data.keys():
            prediction = wordnet.synsets(lemma)[0]
            return prediction

        if type(self.models[lemma]) == type(SVC()):
            context = self.count_vector[lemma].transform([context])
            prediction = self.models[lemma].predict(context)
            prediction = self.label_encoders[lemma].inverse_transform(prediction)
        else:
            prediction = self.models[lemma]
        return prediction

    def evaluate(self, test_dataset, dev_key):
        y_cap = {}
        for key in test_dataset:
            y_cap[key] = self.predict(key, test_dataset)

        correct_prediction = 0
        total_prediction = 0
        for key in y_cap.keys():
            correct_label = wordnet.synset_from_sense_key(dev_key[key][0])
            if y_cap[key] == correct_label:
                correct_prediction = correct_prediction + 1
            total_prediction = total_prediction + 1
        accuracy = correct_prediction / total_prediction
        return round(accuracy, 3)

    def addExtendData(self, dataset):
        for element in dataset:
            lemma = element[0]
            context = element[1]
            correct_label =  element[2]
            if lemma not in self.data.keys():
                self.data[lemma] = {"context": [context], "sense": [correct_label]}
            else:
                self.data[lemma]["context"].append(context)
                self.data[lemma]["sense"].append(correct_label)

    def sample(self, key, instance, instances_keys):

        lemma = instance[key].lemma.decode()
        context = b" ".join(instance[key].context).decode()
        sense = self.predict(key, instance)
        correct_sense_array = []
        for labels in instances_keys[key]:
            correct_sense_array.append(wordnet.synset_from_sense_key(labels))

        print("Target: " + lemma)
        print("Context: " + context)
        for labels in correct_sense_array:
            print("Correct definition: " + labels.definition())
        print("Predicted definition: " + sense.definition())


def load_bootstrap_data(file):
    f = open(file, "r")
    data = json.loads(f.read())
    f.close()
    return data


def label_boot_data(boot_data, lesk_if):
    extended_dataset = []
    for lemma in boot_data.keys():
        for context in boot_data[lemma]:
            if lemma not in context.split():
                continue
            if lesk_if:
                synsets = wordnet.synsets(lemma)
                sense = lesk(context, lemma, synsets=synsets)
            else:
                synsets = wordnet.synsets(lemma)
                sense = synsets[0]
            extended_dataset.append((lemma, context, sense))
    return extended_dataset


if __name__ == "__main__":
    dev_instances, test_instances, dev_key, test_key = loader.load_dataset(INSTANCE_FILE, KEY_FILE)
    boot_data = load_bootstrap_data(BOOTSTRAP)
    extend_dataset = label_boot_data(boot_data, True)
    svm = wsd_SVM(dev_instances, test_instances, dev_key, test_key)
    svm.addExtendData(extend_dataset)
    svm.build_model(dev_instances, test_instances, dev_key, test_key)
    svm.train()
    svm_accuracy = svm.evaluate(test_instances, test_key)
    print(svm_accuracy)
    # svm.sample(SAMPLE_KEY, test_instances, test_key)

