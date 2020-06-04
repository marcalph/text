#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import re
import tensorflow as tf
from sklearn.svm import LinearSVC
from src.utils import load_google_use
import pandas as pd
#todo classification report and shap/eli5 + confusion matrix


def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def load_imdb():
    train_df = load_(os.path.join('/home/malphonsus/.keras/datasets/', "aclImdb", "train"))
    test_df = load_(os.path.join('/home/malphonsus/.keras/datasets/', "aclImdb", "test"))
    return train_df, test_df


train_df, test_df = load_imdb()
print(train_df.head())


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


svm_baseline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('model', LinearSVC())
])


svm_baseline.fit(train_df.sentence, train_df.polarity)
svm_predictions = svm_baseline.predict(test_df.sentence)
print("SVM Accuracy:", np.mean(svm_predictions == test_df.polarity))

guse = load_google_use()
train_embeddings = []
for chunk in chunks(train_df.sentence, 1000):
    print(type(chunk))
    train_embeddings.append(guse(chunk))
train_embeddings = np.vstack(train_embeddings)
print(train_embeddings.shape)
print("encoding for train data done")
test_embeddings = []
for chunk in chunks(train_df.sentence, 1000):
    print(type(chunk))
    test_embeddings.append(guse(chunk))
test_embeddings = np.vstack(test_embeddings)
print(test_embeddings.shape)
print("encoding for train data done")
svm_guse = LinearSVC().fit(train_embeddings, train_df.polarity)
svm_guse_predictions = svm_guse.predict(test_embeddings)
print("SVM guse Accuracy:", np.mean(svm_guse_predictions == test_df.polarity))


