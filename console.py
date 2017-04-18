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
from logic import *
from GMeans import GMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib

from scipy.cluster.hierarchy import ward, dendrogram

parser = argparse.ArgumentParser(description='Cluster a text file with a list of strings')
parser.add_argument('filename', metavar='FILENAME',
                    help='File name with extention that contains the list of strings')

parser.add_argument('--maxC', dest='maxC', type=float, default=0.3,
                    help='add a max commonality parameter for the vectorizer. (default: 0.3). Larger commonality groups things into bigger/more generic clusters')

parser.add_argument('--minC', dest='minC', type=float, default=0.001,
                    help='add a min commonality parameter for the vectorizer. (default: 0.001). Larger commonality eliminates more noise, but may miss some specific clusters.')

parser.add_argument('--namemode', dest='namemode', action='store_const',
                    const=sum,
                    help='just output the most popular tag')

parser.add_argument('--tagmode', dest='tagmode',type=int, default=-1,
                    help='just output the most popular tags')

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
                    help='dry run. Doesnt produce an output file.')

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
tagmode = args.tagmode
dry_run = args.dry

files = []

if os.path.isdir(fileordirname):
    files = [join(fileordirname, f) for f in listdir(fileordirname) if isfile(join(fileordirname, f))]
    files = [f for f in files if not output_file_ext in f]
else:
    files.append(fileordirname)

for filename in files:
    output_file = {}
    
    output_file_name = filename + output_file_ext

    if (not dry_run) and output_file_name:
       output_file = open(output_file_name, 'w')

    print("Processing: " + filename)

    names = []
    infile = codecs.open(filename, encoding='utf-8')
    for line in infile:
        names.append(line.strip())
    infile.close()

    results = process_strings(names, verbose, max_commonality, min_commonality, max_features, ngram_number, kmeans_cluster_count, tag_reduce, namemode, tagmode, 5)

    if not dry_run:
        output_file.write(json.dumps(results))
        output_file.close()