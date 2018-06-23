# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import pickle
from string import punctuation

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences


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
def clean_sentence(sentence, vocab):
    # lower the sentence
    sentence.lower()
    # split into tokens by white space
    tokens = sentence.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)

# load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights("model.h5")

# Get the input length of the models embedding layer
max_length = model.layers[0].get_output_at(0).get_shape().as_list()[1]

# Setup test sentence
testSentence = load_doc("testdata.txt")

# Remove all tokens which are not known to the model
cleanedTestSentence = clean_sentence(testSentence, vocab)

# Creates sequences of the trainingsdata
sequencedTrainingSentence = tokenizer.texts_to_sequences([cleanedTestSentence])

# Pad the sequences to the input_length of the embedding layer
padedTrainingSentenceSequence = pad_sequences(sequencedTrainingSentence, maxlen=max_length)

# Make a prediction on the testsentence
prediction = model.predict(padedTrainingSentenceSequence)

# Print out the prediction
print("Prediction \n", prediction)