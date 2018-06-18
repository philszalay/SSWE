# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from os import listdir
from string import punctuation


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")

print("Loaded model from disk")

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load the test data and process it
testDataFolder = 'testfolder'
testdataDoc = load_doc(testDataFolder)
test_docs = process_docs(testdataDoc, vocab, True)

# create the tokenizer
tokenizer = Tokenizer()

# fit the tokenizer on the test data
tokenizer.fit_on_texts(test_docs)

# sequence encode
sequences = tokenizer.texts_to_sequences(test_docs)

# tokenizer.transform() ??


# Das hier sollte funktionieren.
max_length = max([len(s.split()) for s in test_docs])
print('Found %s unique tokens.' % max_length)

string = "I am a cat"
query = tokenizer.texts_to_sequences(string)
query = pad_sequences(query, maxlen=max_length)

prediction = model.predict_classes(query)

print(prediction)

# label = encoder.inverse_transform(prediction)

# print(label)
