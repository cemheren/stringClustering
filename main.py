from __future__ import print_function
import pickle
import json
import codecs
import sys
import os
import operator
import argparse
import glob
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from collections import OrderedDict
from WordsToNumbers import *
from GMeans import GMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

parser = argparse.ArgumentParser(description='Cluster a text file with a list of strings')
parser.add_argument('filename', metavar='FILENAME',
                    help='File name with extention that contains the list of strings')

parser.add_argument('--maxC', dest='maxC', type=float, default=0.3,
                    help='add a max commonality parameter for the vectorizer. (default: 0.3). Larger commonality groups things into bigger/more generic clusters')

parser.add_argument('--minC', dest='minC', type=float, default=0.001,
                    help='add a min commonality parameter for the vectorizer. (default: 0.001). Larger commonality eliminates more noise, but may miss some specific clusters.')

parser.add_argument('--noeng', dest='noenglish', action='store_const',
                    const=sum,
                    help='option to disable English dictionary and use contextual dictionary')

parser.add_argument('--namemode', dest='namemode', action='store_const',
                    const=sum,
                    help='just output the most popular tag')

parser.add_argument('--ngram', dest='ngram', type=int, default=1,
                    help='number of tokens to take into account when calculating the distance matrix. Pick a larger number if your data is diverse')

parser.add_argument('--tagreduce', dest='tagreduce', action='store_const',
                    const=sum,
                    help='Cleverly reduces number of produced tags')

parser.add_argument('--clusters', dest='clusterCount', type=int, default=-1,
                    help='Number of clusters to create (default: use cmeans to autodetect)')

parser.add_argument('--features', dest='maxfeatures', type=int, default=100,
                    help='Number features for the tfidf vectorizer. Increase it when your data is very diverse')

parser.add_argument('--out', dest='outputextention', default=".cls", help='output file extention')

parser.add_argument('--verbose', dest='verbose', action='store_const',
                    const=sum,
                    help='Verbose output')

parser.add_argument('--dry', dest='dry', action='store_const',
                    const=sum,
                    help='Verbose output')

args = parser.parse_args()

fileordirname = args.filename
max_commonality = args.maxC
min_commonality = args.minC
output_file_ext = args.outputextention
verbose = args.verbose
tag_reduce = args.tagreduce
ngram_number = args.ngram
max_features = args.maxfeatures
kmeans_cluster_count = args.clusterCount
namemode = args.namemode
dry_run = args.dry

top_level_excluded_terms = ["aa", "oi"]

files = []

if os.path.isdir(fileordirname):
    files = [join(fileordirname, f) for f in listdir(fileordirname) if isfile(join(fileordirname, f))]
else:
    files.append(fileordirname)

for filename in files:
    output_file = {}
    results = {}
    output_file_name = filename + output_file_ext

    if (not dry_run) and output_file_name:
       output_file = open(output_file_name, 'w')

    print("Processing: " + filename)

    names = []
    infile = codecs.open(filename, encoding='utf-8')
    for line in infile:
        names.append(line.strip())
    infile.close()

    names_split = [(split_words_in_text(n)) for n in names]
    # print(names_split)

    if verbose:
        print("vectorizing...")
    
    tfidf_vectorizer = TfidfVectorizer(max_df=max_commonality, max_features=max_features,
                                    min_df=min_commonality, tokenizer=split_words_in_text,
                                    use_idf=True, ngram_range=(1, ngram_number))
                                    
    tfidf_matrix = tfidf_vectorizer.fit_transform(names)  # fit the vectorizer to synopses

    if verbose:
        print(tfidf_matrix.shape)

    terms = tfidf_vectorizer.get_feature_names()
    if output_file:
        results["terms_used_to_cluster"] = terms
    
    if verbose:
        print(terms)
        print()
        print()

    excluded_terms = []

    try:
        tfidf_vectorizer2 = TfidfVectorizer(max_df=0.9999, max_features=max_features,
                                        min_df=max_commonality, tokenizer=split_words_in_text,
                                        use_idf=True, ngram_range=(1, ngram_number))

        tfidf_matrix2 = tfidf_vectorizer2.fit_transform(names)  # fit the vectorizer to synopses
        excluded_terms = tfidf_vectorizer2.get_feature_names()
        if output_file:
            results["excluded_terms"] = excluded_terms
        
        if verbose:
            print(excluded_terms)
            print()
            print()
    except: 
        z = 5


    if kmeans_cluster_count == -1:
        gmeans = GMeans(random_state=1010,
                strictness=0)
        gmeans.fit(tfidf_matrix)
        mean_clusters = gmeans.labels_
    else:
        km = KMeans(n_clusters=kmeans_cluster_count)
        km.fit(tfidf_matrix)
        mean_clusters = km.labels_.tolist()

    pandas_helper = {'names': names, 'cluster': mean_clusters}
    frame = pd.DataFrame(pandas_helper, index=[mean_clusters])
    frame['cluster'].value_counts()

    groups = frame.groupby('cluster')

    clusters = []
    all_tags = dict()
    for name, group in groups:

        result_group = {}
        values = group.values[:,1]
        tags = dict()
        for v in values:
            split_value = split_words_in_text(v)
            for sv in split_value:
                if sv in tags:
                    tags[sv] = tags[sv] + 1
                else:
                    if sv not in excluded_terms and len(sv) > 1:
                        tags[sv] = 1

        sorted_tags = list(reversed(sorted(tags.items(), key=operator.itemgetter(1))))
        important_tags = list()
        threshold = 10
        if tag_reduce:
            threshold = 2

        i = 0
        for t in sorted_tags:
            i += 1

            if tag_reduce:
                if t[1] >= (len(values) / 2):
                    important_tags.append(t[0])
            else:
                if t[1] >= 1:
                    important_tags.append(t[0])
            
            if i > threshold:
                break

        for tag in important_tags:
            if tag in all_tags:
                a = 5
            else:
                all_tags[tag] = list()

        if verbose:
            print(len(values))
            print(values)
        
        if output_file:
            result_group["values"] = list(values)
            result_group["tags"] = sorted_tags

        clusters.append(result_group)
    
    if verbose:
        print("----------------------------------------------------------")

    results["clusters"] = clusters

    i = 0
    for name_list in names_split:
        for tag in all_tags:
            if tag in name_list:
                all_tags[tag].append(names[i])
        i += 1

    return_tags = dict()

    local_tag_threshold = 1
    if namemode:
        local_tag_threshold = 0

    for tag in all_tags.keys():
        if len(all_tags[tag]) > local_tag_threshold:
            return_tags[tag] = all_tags[tag]

    if verbose:
        print(return_tags.keys())

    if output_file or dry_run:
        return_tags = OrderedDict(sorted(return_tags.items(), key=lambda x: len(x[1]))) 

        if namemode:
            max_key = ""
            max_value = 0

            while(len(return_tags) > 0):
                key, value = return_tags.popitem()

                if key in top_level_excluded_terms:
                    continue

                if max_value < len(value):
                    max_value = len(value)
                else:
                    break

                if len(key) > len(max_key):
                    max_key = key

            results = max_key
            print(max_key)
        else:
            return_tags = OrderedDict(sorted(return_tags.items()))

            if tag_reduce:
                
                if verbose:
                    print("reducing number of tags...")

                tags_to_remove = {}

                for key1 in return_tags.keys():
                    for key2 in return_tags.keys():
                        if key1 != key2 and key1 in key2:
                            set1 = set(return_tags[key1])
                            set2 = set(return_tags[key2])

                            if set1<=set2:
                                tags_to_remove[key1] = 0
                            if set2<set1:
                                tags_to_remove[key2] = 0                           

                for t in tags_to_remove.keys():
                    del return_tags[t]

            results["tags"] = return_tags

        if not dry_run:
            output_file.write(json.dumps(results))
            output_file.close()