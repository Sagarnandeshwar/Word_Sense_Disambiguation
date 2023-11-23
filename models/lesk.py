from nltk.wsd import lesk
from nltk.corpus import wordnet

import loader

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

lemmatizer = WordNetLemmatizer()


STOPWORDS = set(stopwords.words('english'))

INSTANCE_FILE = '../dataset/multilingual-all-words.en.xml'
KEY_FILE = '../dataset/wordnet.en.key'
SAMPLE_KEY = "d008.s027.t002"


def process_contecxt(context, position):
    word_token = word_tokenize(context)
    pos = pos_tag(word_token)

    new_sentence = []

    for words in word_token:
        if not words.isalpha():
            continue
        if words in STOPWORDS:
            continue

        words = words.lower()
        words = lemmatizer.lemmatize(words)
        new_sentence.append(words)

    new_sentence = TreebankWordDetokenizer().detokenize(new_sentence)
    return new_sentence, pos[position][1].lower()


def eval_lesk(instances, instances_keys):
    lesk_correct = 0
    total_prediction = 0

    for key in instances:
        context = b" ".join(instances[key].context).decode()
        lemma = instances[key].lemma.decode()
        position = instances[key].index

        context, POS = process_contecxt(context, position)

        if POS[0].lower in ["n", "a", "v", "s"]:
            POS = POS[0].lower()
        else:
            POS = "n"

        synsets = wordnet.synsets(lemma)
        lesk_sense = lesk(context, lemma,  pos=POS, synsets=synsets)
        correct_sense_array = []
        for labels in instances_keys[key]:
            correct_sense_array.append(wordnet.synset_from_sense_key(labels))

        if lesk_sense in correct_sense_array:
            lesk_correct = lesk_correct + 1

        total_prediction = total_prediction + 1
    lesk_accuracy = lesk_correct / total_prediction
    return round(lesk_accuracy,3)


def sample_output(key, instances, instances_keys):
    context = b" ".join(instances[key].context).decode()
    lemma = instances[key].lemma.decode()
    position = instances[key].index

    context, POS = process_contecxt(context, position)

    if POS[0].lower in ["n", "a", "v", "s"]:
        POS = POS[0].lower()
    else:
        POS = "n"

    synsets = wordnet.synsets(lemma)
    # lesk_sense = lesk(context, lemma, pos=POS, synsets=synsets)
    lesk_sense = lesk(context, lemma, synsets=synsets)
    correct_sense_array = []
    for labels in instances_keys[key]:
        correct_sense_array.append(wordnet.synset_from_sense_key(labels))

    print("Target: " + lemma)
    print("Context: " + context)
    for labels in correct_sense_array:
        print("Correct definition: " + labels.definition())
    print("Predicted definition: " + lesk_sense.definition())


if __name__ == "__main__":
    dev_instances, test_instances, dev_key, test_key = loader.load_dataset(INSTANCE_FILE, KEY_FILE)
    lesk_dev_accuracy = eval_lesk(dev_instances, dev_key)
    lesk_test_accuracy = eval_lesk(test_instances, test_key)
    print(lesk_dev_accuracy)
    print(lesk_test_accuracy)
    # sample_output(SAMPLE_KEY, test_instances, test_key)
