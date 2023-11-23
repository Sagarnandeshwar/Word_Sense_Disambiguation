import loader
import numpy as np
from keytotext import pipeline
import json


INSTANCE_FILE = 'dataset/multilingual-all-words.en.xml'
KEY_FILE = 'dataset/wordnet.en.key'

if __name__ == "__main__":
    words = []
    dev_instances, test_instances, dev_key, test_key = loader.load_dataset(INSTANCE_FILE, KEY_FILE)
    for key in dev_instances.keys():
        lemma = dev_instances[key].lemma.decode()
        words.append(lemma)
    for key in test_instances.keys():
        lemma = test_instances[key].lemma.decode()
        words.append(lemma)

    words = np.unique(words)

    nlp = pipeline("k2t")
    config = {"do_sample": True, "no_repeat_ngram_size": 10}

    json_dict = {}

    for index in range(len(words)):
        w = words[index]
        w_list = []
        for i in range(10):
            w_list.append(nlp(w, **config))
        json_dict[w] = w_list
        print(index)

    with open('../dataset/data.json', 'w') as f:
        json.dump(json_dict, f)






