from __future__ import print_function
import numpy as np
import pickle
import pandas as pd
import json
from WordsToNumbers import *
import matplotlib.pyplot as plt
import codecs

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib

from scipy.cluster.hierarchy import ward, dendrogram

def split_words_in_text(text):
    split = list()
    split = sentence_to_word_array(text, True, False, True)

    if len(split) > 0:
        return split

    split.append(text.lower())

    return split

# graph = open('myGraph.txt', 'r')
# graph = json.load(graph)
# names = [n['Name'] for n in graph]

names = []
infile = codecs.open('omsProd.txt', encoding='utf-8')
for line in infile:
    names.append(line.strip())
infile.close()

# names_split = [(split_words_in_text(n)) for n in names]

print("vectorizing...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.001, max_features=200000,
                                 min_df=0.0001, tokenizer=split_words_in_text,
                                 use_idf=True, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(names)  # fit the vectorizer to synopses

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
# print(terms)

dist = 1 - cosine_similarity(tfidf_matrix)

print("clustering...")
for c in range(30, 31):
    num_clusters = c
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    ss = silhouette_score(tfidf_matrix, km.labels_, metric='euclidean')
    print("silhouette_score: , num_clusters:", ss, num_clusters)

# uncomment the below to save your model
# since I've already run my model I am loading from the pickle
# joblib.dump(km,  'doc_cluster.pkl')
# km = joblib.load('doc_cluster.pkl')

names = {'names': names, 'cluster': clusters}
frame = pd.DataFrame(names, index=[clusters])

frame['cluster'].value_counts()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

groups = frame.groupby('cluster')

for name, group in groups:
    raw_input("Press Enter to continue...")
    print(len(group.values))
    print(group.values)

a = 5