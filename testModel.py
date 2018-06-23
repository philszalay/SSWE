# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import os
import pickle
from string import punctuation

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf-8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_review(filepath):
    # fetch directory path
    script_dir = os.path.dirname(__file__)
    # join absolute with relative filepath
    abs_file_path = os.path.join(script_dir, filepath)
    # load file
    review = load_doc(abs_file_path)
    # return the file
    return review


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


def rating_from_filename(filename):
    splitted_filename = filename.split('_')
    rating_with_file_ending = splitted_filename[2]
    rating = rating_with_file_ending.split('.')[0]
    return rating


def sentiment_from_filename(filename):
    return filename.split('/')[1]


def predict_review(review):
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

    # Remove all tokens which are not known to the model
    cleaned_test_sentence = clean_sentence(review, vocab)

    # Creates sequences of the trainingsdata
    sequenced_training_sentence = tokenizer.texts_to_sequences([cleaned_test_sentence])

    # Pad the sequences to the input_length of the embedding layer
    paded_training_sentence_sequence = pad_sequences(sequenced_training_sentence, maxlen=max_length)

    # Make a prediction on the testsentence
    prediction = model.predict(paded_training_sentence_sequence)

    # Return the result
    return prediction


def predict_metacritic_review(filepath):
    # Load Review
    review = load_review(filepath)
    # Predict the sentiment of the review
    prediction = predict_review(review)
    # Print out the prediction
    print("Prediction of metacritic-review with rating", rating_from_filename(filepath), ":", prediction)


def predict_metacritic_user_review(filepath):
    # Load Review
    review = load_review(filepath)
    # Predict the sentiment of the review
    prediction = predict_review(review)
    # Print out the prediction
    print("Prediction of metacritic user review with rating", rating_from_filename(filepath), ":", prediction)


def predict_testset_review(filepath):
    # Load Review
    review = load_review(filepath)
    # Predict the sentiment of the review
    prediction = predict_review(review)
    print("Prediction of review with sentiment", sentiment_from_filename(filepath), ":", prediction)


# Predict reviews from http://www.metacritic.com/movie -> Rating between 0 and 100
predict_metacritic_review("metacriticReviews/01_review_30.txt")
predict_metacritic_review("metacriticReviews/02_review_100.txt")

# Predict user reviews from http://www.metacritic.com/movie -> Rating between 0 and 10
predict_metacritic_user_review("metacriticUserReviews/01_userreview_2.txt")
predict_metacritic_user_review("metacriticUserReviews/02_userreview_4.txt")

# Predict reviews from the testset -> Rating between negative and positive
predict_testset_review("txt_sentoken/neg/cv000_29416.txt")
predict_testset_review("txt_sentoken/neg/cv001_19502.txt")
predict_testset_review("txt_sentoken/neg/cv002_17424.txt")
predict_testset_review("txt_sentoken/neg/cv115_26443.txt")
predict_testset_review("txt_sentoken/pos/cv000_29590.txt")
predict_testset_review("txt_sentoken/pos/cv009_29592.txt")
