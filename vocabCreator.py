from os import listdir
from string import punctuation

from nltk.corpus import stopwords


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf-8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def add_to_vocab(vocab, tokens):
    for t in tokens:
        vocab.append(t)


def create_vocab_from_tweets(directory):
    vocab = []
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if not filename.startswith('tweet'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc)
        # add tokens to vocab list
        add_to_vocab(vocab, tokens)

    return vocab


def remove_unwantedtokens(vocab, min_occurance):
    # keep tokens with a min occurrence
    tokens = [k for k in vocab if vocab.count(k) >= min_occurance]

    return tokens


twitter_pos = create_vocab_from_tweets("twitterdata/pos")
twitter_neg = create_vocab_from_tweets("twitterdata/neg")

twitter_vocab = []
twitter_vocab.extend(twitter_pos)
twitter_vocab.extend(twitter_neg)

file = "twitter_vocab.txt"

twitter_vocab = remove_unwantedtokens(twitter_vocab, 2)

with open(file, 'a', encoding="utf-8") as f:
    for token in twitter_vocab:
        f.write(str(token) + "\n")
    print("Anzahl an Tokens \n:", len(twitter_vocab))
    print(file, "successfully created.")
