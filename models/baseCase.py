from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import loader

INSTANCE_FILE = '../dataset/multilingual-all-words.en.xml'
KEY_FILE = '../dataset/wordnet.en.key'

SAMPLE_KEY = "d003.s003.t001"


def process_contecxt(context, position):
    word_token = word_tokenize(context)
    pos = pos_tag(word_token)
    return pos[position][1].lower()


def eval_baseCase(instances, instances_keys):
    baseCase_correct = 0
    total_prediction = 0

    for key in instances:
        lemma = instances[key].lemma.decode()

        context = b" ".join(instances[key].context).decode()
        position = instances[key].index

        POS = process_contecxt(context, position)
        if POS[0].lower in ["n", "a", "v", "s"]:
            POS = POS[0].lower()
        else:
            POS = "n"

        synsets = wordnet.synsets(lemma, pos=POS)
        baseCase_sense = synsets[0]
        correct_sense_array = []
        for labels in instances_keys[key]:
            correct_sense_array.append(wordnet.synset_from_sense_key(labels))

        if baseCase_sense in correct_sense_array:
            baseCase_correct = baseCase_correct + 1

        total_prediction = total_prediction + 1

    baseCase_accuracy = baseCase_correct / total_prediction
    return round(baseCase_accuracy,3)


def sample_output(key, instances, instances_keys):
    lemma = instances[key].lemma.decode()
    context = b" ".join(instances[key].context).decode()

    position = instances[key].index
    POS = process_contecxt(context, position)
    if POS[0].lower in ["n", "a", "v", "s"]:
        POS = POS[0].lower()
    else:
        POS = "n"

    synsets = wordnet.synsets(lemma, pos=POS)
    baseCase_sense = synsets[0]
    correct_sense_array = []
    for labels in instances_keys[key]:
        correct_sense_array.append(wordnet.synset_from_sense_key(labels))

    print("Target: " + lemma)
    print("Context: " + context)
    for labels in correct_sense_array:
        print("Correct definition: " + labels.definition())
    print("Predicted definition: " + baseCase_sense.definition())



if __name__ == "__main__":
    dev_instances, test_instances, dev_key, test_key = loader.load_dataset(INSTANCE_FILE, KEY_FILE)
    baseCase_dev_accuracy = eval_baseCase(dev_instances, dev_key)
    baseCase_test_accuracy = eval_baseCase(test_instances, test_key)

    print(baseCase_dev_accuracy)
    print(baseCase_test_accuracy)

    sample_output(SAMPLE_KEY, test_instances, test_key)



