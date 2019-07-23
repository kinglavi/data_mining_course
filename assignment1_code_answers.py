# Question 4 -


# --- TASK 2
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Data Loading:
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('https://raw.githubusercontent.com/m-braverman/ta_dm_course_data/master/train3.csv')

text_column = 'review_text'

# df[f"tokenized_{text_column}"] = df.apply(lambda row: word_tokenize(row[text_column]), axis=1)

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("something")

stopwords = set(nltk.corpus.stopwords.words('english') + ['reuter', '\x03'])


def lemmatizer_and_lower_udf(row, column_name):
    tokenized_row = row[column_name]

    new_array_of_words = []
    for word in tokenized_row:
        word = word.lower()
        if word not in stopwords:
            new_array_of_words.append(lemmatizer.lemmatize(word))

    return new_array_of_words


df[f"lemmatized_{text_column}"] = df.apply(lambda row: lemmatizer_udf(row, f"tokenized_{text_column}"), axis=1)

print(df)
# sent_tokenize(df, 'english')

# from collections import Counter
#
# import pandas as pd
#
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
#
# plt.style.use('ggplot')
#
# import numpy as np
#
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
#
# from sklearn.preprocessing import StandardScaler
#
# df = pd.read_csv('https://raw.githubusercontent.com/m-braverman/ta_dm_course_data/master/x_y_terrorism_data.csv')
# print("dfdfd")

# ----------- TASK 1 ------------
# df_std = StandardScaler().fit_transform(df)
# plt.scatter(df_std[:, 1], df_std[:, 0])
#
# clustering = DBSCAN(eps=3, min_samples=2).fit(df_std)
#
# # Black removed and is used for noise instead.
# core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
# core_samples_mask[clustering.core_sample_indices_] = True
# labels = clustering.labels_
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = df_std[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#
#     xy = df_std[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
#
# plt.title('Cluster is - ')
# plt.show()

# ns = 4
# nbrs = NearestNeighbors(n_neighbors=ns).fit(df_std)
# distances, indices = nbrs.kneighbors(df_std)
# distanceDec = sorted(distances[:,ns-1], reverse=True)
#
# plt.plot(indices[:,0], distanceDec)
#
# clustering = KMeans(n_clusters=21).fit(df_std)
# clustering.


# def pre_process(row, column_name):
#     text_column = row[column_name]
#     text_column = text_column.translate(table)
#     text_column = re.sub(r'\d+', 'num', text_column)
#
#     tokenized_row = word_tokenize(text_column)
#
#     # Lemmatize + remove stop words + lower
#     new_array_of_words = []
#     for word in tokenized_row:
#         word = word.lower()
#         if word not in stopwords:
#             new_array_of_words.append(lemmatizer.lemmatize(word))
#
#     text_column = " ".join(new_array_of_words)
#
#     row[column_name] = text_column
#
#     return row

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
df

# Counter
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_text_counts = count_vect.fit_transform(df[f"final_{text_column}"])


# Convert to feature vector
tf_counter = TfidfVectorizer(max_features = 1000)
tfidfX = tf_counter.fit_transform(train[f"final_{text_column}"])

tfidfX