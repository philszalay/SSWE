import pickle
from os import listdir
from string import punctuation

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers.convolutional import AveragePooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import array


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf-8")
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
def process_docs(directory, vocab, is_training_data):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # In case we only want to process our training tweets
        if is_training_data and not filename.startswith("tweet"):
            continue
        # In case we only want to process our test tweets
        if not is_training_data and filename.startswith("tweet"):
            continue
        # skip any reviews in the test set
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents


def save_model(model, tokenizer):
    # save Tokenizer to file
    with open('twitter_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # serialize model to JSON
    model_json = model.to_json()
    with open("twitterSentimentClassficationModel.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("twitterSentimentClassficationWeights.h5")
    print("Saved model to disk")


# load the vocabulary
vocab_filename = 'twitter_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('twitterdata/pos', vocab, True)
negative_docs = process_docs('twitterdata/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()

# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
xTrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
yTrain = array([0 for _ in range(5000)] + [1 for _ in range(5000)])

# load all test reviews
positive_docs = process_docs('twitterdata/pos', vocab, False)
negative_docs = process_docs('twitterdata/neg', vocab, False)

test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)

# pad sequences
xTest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define test labels 0 = negative, 1 = positive
yTest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
# kernel_size -> defines the size of the sliding window;
# filters -> how many different windows with the size of the kernel_size
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='relu'))  # Dense: Fully connected layer
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(xTrain, yTrain, epochs=3, verbose=2)

# evaluate
loss, acc = model.evaluate(xTest, yTest, verbose=0)

print('Test Accuracy: %f' % (acc * 100))

save_model(model, tokenizer)
