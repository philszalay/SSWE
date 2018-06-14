# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")

print("Loaded model from disk")

# load label names to use in prediction results
testphrase = "Der Schauspieler hat eine gute Arbeit geleistet"

# load the vocabulary
vocab = testphrase.split()
vocab = set(vocab)

# load all training reviews
train_docs = vocab

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
XTest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

testIt = numpy.array(XTest)

# predict the probability across all output classes
y_proba = model.predict(XTest)

print(y_proba)