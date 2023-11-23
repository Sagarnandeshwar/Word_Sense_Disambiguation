from sklearn.tree import DecisionTreeClassifier
import loader
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json

from nltk.wsd import lesk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import random

STOPWORDS = stopwords.words('english')
ps = PorterStemmer()
wl = WordNetLemmatizer()

INSTANCE_FILE = '../dataset/multilingual-all-words.en.xml'
KEY_FILE = '../dataset/wordnet.en.key'
BOOTSTRAP = '../dataset/data.json'
SAMPLE_KEY = "d003.s002.t010"

class wsd_DecisionTree():
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
            self.models[key] = DecisionTreeClassifier()
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
                lesk_algo = random.randint(0, 1)
                if lesk_algo:
                    synsets = wordnet.synsets(lemma)
                    sense = lesk(lemma_context_org[0], lemma, synsets=synsets)
                else:
                    synsets = wordnet.synsets(lemma)
                    sense = synsets[0]
                self.models[lemma] = sense
                # self.models[lemma] = lemma_sense[0]

            else:
                self.models[lemma] = DecisionTreeClassifier()
                self.models[lemma].fit(X, Y)

    def predict(self, key, test_dataset):
        lemma = test_dataset[key].lemma.decode()
        context = b" ".join(test_dataset[key].context).decode()

        if lemma not in self.data.keys():
            prediction = wordnet.synsets(lemma)[0]
            return prediction

        if type(self.models[lemma]) == type(DecisionTreeClassifier()):
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
            correct_label = element[2]
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


    def bootstrap3(self, boot_data):
        for lemma in boot_data.keys():
            if lemma not in self.data.keys():
                context_array = []
                context_label_array = []
                self.models[lemma] = DecisionTreeClassifier()
                self.count_vector[lemma] = CountVectorizer()
                self.label_encoders[lemma] = LabelEncoder()

                for context in boot_data[lemma]:

                    lesk_algo = random.randint(0, 1)
                    if lesk_algo:
                        synsets = wordnet.synsets(lemma)
                        sense = lesk(context, lemma, synsets=synsets)
                    else:
                        synsets = wordnet.synsets(lemma)
                        sense = synsets[0]

                    context_array.append(context)
                    context_label_array.append(sense)

                if len(context_array) == 0:
                    continue

                self.data[lemma] = {"context": context_array, "sense": context_label_array}
                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)
                continue

            context_label_array = []
            context_array = []
            for context in boot_data[lemma]:
                context_array.append(context)
                if type(self.models[lemma]) == type(DecisionTreeClassifier()):
                    context = self.count_vector[lemma].transform([context])
                    context_label = self.models[lemma].predict(context)
                    context_label = self.label_encoders[lemma].inverse_transform(context_label)[0]
                    context_label_array.append(context_label)
                else:
                    lesk_algo = random.randint(0, 1)
                    if lesk_algo:
                        synsets = wordnet.synsets(lemma)
                        sense = lesk(context, lemma, synsets=synsets)
                    else:
                        synsets = wordnet.synsets(lemma)
                        sense = synsets[0]
                    context_label_array.append(sense)

            if type(self.models[lemma]) == type(DecisionTreeClassifier()):
                for label in context_label_array:
                    self.data[lemma]["sense"].append(label)
                for context in context_array:
                    self.data[lemma]["context"].append(context)

                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)

            else:
                for label in context_label_array:
                    self.data[lemma]["sense"].append(label)
                for context in context_array:
                    self.data[lemma]["context"].append(context)

                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)

    def bootstrap2(self, boot_data):
        for lemma in boot_data.keys():
            if lemma not in self.data.keys():
                context_array = []
                context_label_array = []

                self.models[lemma] = DecisionTreeClassifier()
                self.count_vector[lemma] = CountVectorizer()
                self.label_encoders[lemma] = LabelEncoder()

                for context in boot_data[lemma]:

                    lesk_algo = random.randint(0, 1)
                    if lesk_algo:
                        synsets = wordnet.synsets(lemma)
                        sense = lesk(context, lemma, synsets=synsets)
                        score = 0.4
                    else:
                        synsets = wordnet.synsets(lemma)
                        sense = synsets[0]
                        score = 0.6

                    if score > 0.5:
                        context_array.append(context)
                        context_label_array.append(sense)

                if len(context_array) == 0:
                    continue

                self.data[lemma] = {"context": context_array, "sense": context_label_array}
                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)
                continue

            context_label_array = []
            context_array = []
            for context in boot_data[lemma]:
                c = context
                if type(self.models[lemma]) == type(DecisionTreeClassifier()):
                    context = self.count_vector[lemma].transform([context])
                    sense = self.models[lemma].predict(context)
                    prob = self.models[lemma].predict_proba(context)
                    score = prob[0][sense[0]]
                    sense = self.label_encoders[lemma].inverse_transform(sense)[0]
                else:
                    lesk_algo = random.randint(0, 1)
                    if lesk_algo:
                        synsets = wordnet.synsets(lemma)
                        sense = lesk(context, lemma, synsets=synsets)
                        score = 0.4
                    else:
                        synsets = wordnet.synsets(lemma)
                        sense = synsets[0]
                        score = 0.6
                if score > 0.5:
                    context_array.append(c)
                    context_label_array.append(sense)

            if len(context_array) == 0:
                continue

            if type(self.models[lemma]) == type(DecisionTreeClassifier()):
                for label in context_label_array:
                    self.data[lemma]["sense"].append(label)
                for context in context_array:
                    self.data[lemma]["context"].append(context)

                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)

            else:
                for label in context_label_array:
                    self.data[lemma]["sense"].append(label)
                for context in context_array:
                    self.data[lemma]["context"].append(context)

                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)

    def bootstrap1(self, boot_data):
        for lemma in boot_data.keys():

            if lemma not in self.data.keys():
                context_array = []
                context_label_array = []

                self.models[lemma] = DecisionTreeClassifier()
                self.count_vector[lemma] = CountVectorizer()
                self.label_encoders[lemma] = LabelEncoder()

                for context in boot_data[lemma]:
                    lesk_algo = random.randint(0, 1)
                    if lesk_algo:
                        synsets = wordnet.synsets(lemma)
                        sense = lesk(context, lemma, synsets=synsets)
                        score = 0.4
                    else:
                        synsets = wordnet.synsets(lemma)
                        sense = synsets[0]
                        score = 0.6
                    if score > 0.9:
                        context_array.append(context)
                        context_label_array.append(sense)

                if len(context_array) <= 1:
                    continue

                self.data[lemma] = {"context": context_array, "sense": context_label_array}
                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)
                continue

            context_label_array = []
            context_array = []
            for context in boot_data[lemma]:
                c = context
                if type(self.models[lemma]) == type(DecisionTreeClassifier()):

                    context = self.count_vector[lemma].transform([context])
                    sense = self.models[lemma].predict(context)
                    prob = self.models[lemma].predict_proba(context)
                    score = prob[0][sense[0]]
                    sense = self.label_encoders[lemma].inverse_transform(sense)[0]
                else:
                    lesk_algo = random.randint(0, 1)
                    if lesk_algo:
                        synsets = wordnet.synsets(lemma)
                        sense = lesk(context, lemma, synsets=synsets)
                        score = 0.4
                    else:
                        synsets = wordnet.synsets(lemma)
                        sense = synsets[0]
                        score = 0.6
                if score > 0.9:
                    context_array.append(c)
                    context_label_array.append(sense)

            if len(context_array) <= 1:
                continue

            if type(self.models[lemma]) == type(DecisionTreeClassifier()):
                for label in context_label_array:
                    self.data[lemma]["sense"].append(label)
                for context in context_array:
                    self.data[lemma]["context"].append(context)

                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)

            else:
                for label in context_label_array:
                    self.data[lemma]["sense"].append(label)
                for context in context_array:
                    self.data[lemma]["context"].append(context)

                lemma_context = self.data[lemma]["context"]
                lemma_sense = self.data[lemma]["sense"]

                self.count_vector[lemma].fit(lemma_context)
                self.label_encoders[lemma].fit(lemma_sense)

                X = self.count_vector[lemma].transform(lemma_context)
                Y = self.label_encoders[lemma].transform(lemma_sense)

                if len(np.unique(Y)) <= 1:
                    self.models[lemma] = lemma_sense[0]
                else:
                    self.models[lemma] = DecisionTreeClassifier()
                    self.models[lemma].fit(X, Y)


def load_bootstrap_data(file):
    f = open(file, "r")
    boot_data = json.loads(f.read())
    f.close()
    for lemma in boot_data.keys():
        new_context = []
        for context in boot_data[lemma]:
            if lemma not in context.split():
                continue
            new_context.append(context)
        boot_data[lemma] = new_context
    return boot_data


if __name__ == "__main__":
    dev_instances, test_instances, dev_key, test_key = loader.load_dataset(INSTANCE_FILE, KEY_FILE)
    boot_data = load_bootstrap_data(BOOTSTRAP)

    decisionTree = wsd_DecisionTree(dev_instances, test_instances, dev_key, test_key)
    decisionTree.build_model(dev_instances, test_instances, dev_key, test_key)

    decisionTree.train()
    decisionTree_accuracy = decisionTree.evaluate(test_instances, test_key)
    print(decisionTree_accuracy)

    decisionTree.bootstrap1(boot_data)
    decisionTree_accuracy = decisionTree.evaluate(test_instances, test_key)
    print(decisionTree_accuracy)

    decisionTree.bootstrap2(boot_data)
    decisionTree_accuracy = decisionTree.evaluate(test_instances, test_key)
    print(decisionTree_accuracy)

    decisionTree.bootstrap3(boot_data)
    decisionTree_accuracy = decisionTree.evaluate(test_instances, test_key)
    print(decisionTree_accuracy)

    # decisionTree.sample(SAMPLE_KEY, test_instances, test_key)
