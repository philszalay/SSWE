from os import listdir
from string import punctuation


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
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


twitter_pos = create_vocab_from_tweets("twitterdata/pos")
twitter_neg = create_vocab_from_tweets("twitterdata/neg")

twitter_vocab = []
twitter_vocab.extend(twitter_pos)
twitter_vocab.extend(twitter_neg)

file = "twitter_vocab.txt"

with open(file, 'a', encoding="utf-8") as f:
    for token in twitter_vocab:
        f.write(str(token) + "\n")
    print(file, "successfully created.")
