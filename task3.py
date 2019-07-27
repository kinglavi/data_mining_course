# imports - add additional imports here
import gensim
import numpy
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/m-braverman/ta_dm_course_data/master/train3.csv')

from nltk.tokenize import word_tokenize

# df['tokenized'] = df.apply(lambda row: word_tokenize(row['review_text'].lower()), axis=1)

import nltk
import string
import re
from copy import deepcopy
from nltk.tokenize import word_tokenize, sent_tokenize

# nltk.download() - You must download nltk for using
stopwords = set(nltk.corpus.stopwords.words('english') + ['reuter', '\x03'])
text_column = 'review_text'


def pre_process(row, column_name):
    text_column = deepcopy(row[column_name])

    # Replace numbers wih 'num'
    text_column = re.sub(r'\d+', 'num', text_column)

    # Tokenize
    tokenized_row = word_tokenize(text_column.lower())

    # remove stop words + lower + remove punctuation
    for word in tokenized_row:
        if word in stopwords \
                or word in string.punctuation:
            tokenized_row.remove(word)

    return tokenized_row


df[f"final_{text_column}"] = df.apply(lambda row: pre_process(row, text_column), axis=1)

model = gensim.models.Word2Vec(df["final_review_text"], min_count=5, size=200)

model.wv.most_similar('good')


def calculate_sentence_embedding(tokenized_review_column, word_embed_model):
    tokenized_review_column = deepcopy(tokenized_review_column)
    review_word_vectors = []
    for word in tokenized_review_column:
        if word in word_embed_model.wv.vocab:
            review_word_vectors.append(model[word])

    review_embedding = numpy.average(review_word_vectors, axis=0)
    return review_embedding


df['review_embedding'] = df.apply(
    lambda row: calculate_sentence_embedding(
        row[f"final_{text_column}"], word_embed_model=model),
    axis=1
)
