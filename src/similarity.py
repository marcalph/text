#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################

import functools as ft
from collections import Counter

import nltk
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

from utils import load_glove, load_sts_data, load_infersent, load_google_use

glove = load_glove("data/embeddings/glove.840B/glove.840B.300d.txt")
sts_train = load_sts_data()
sts_dev = load_sts_data("data/stsbenchmark/sts-dev.csv")







# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)
# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/4")


class Text(object):
    def __init__(self, text):
        self.raw = text
        self.tokens = [t.lower() for t in nltk.word_tokenize(self.raw)]
        self.tokens_wosw = [t for t in self.tokens if t not in STOP_WORDS]






def base_sim(text_left, text_right, embedding, wosw=False, tfidf=False):
    """ compute sentence similairty through embeddings either w/wo:
        stopwords
        tf-idf frequencies
    """
    if tfidf:
        tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)
        tfidf_vectorizer = tfidf_vectorizer.fit([t.raw for t in text_left] + [t.raw for t in text_right])
        freq_dict = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
        # print(freq_dict.keys())

    sims = []
    for (textl, textr) in zip(text_left, text_right):
        tokensl = textl.tokens_wosw if wosw else textl.tokens
        tokensr = textr.tokens_wosw if wosw else textr.tokens
        # remove token oov for embedding
        tokensl = [token for token in tokensl if token in embedding]
        tokensr = [token for token in tokensr if token in embedding]
        if len(tokensl) == 0 or len(tokensr) == 0:
            sims.append(0)
            continue

        countsl = Counter(tokensl)
        countsr = Counter(tokensr)
        # print(countsl.keys())

        weights1 = [countsl[token] * freq_dict[token]
                    for token in countsl] if tfidf else None
        weights2 = [countsr[token] * freq_dict[token]
                    for token in countsr] if tfidf else None

        embedding1 = np.average([embedding[token] for token in countsl], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([embedding[token] for token in countsr], axis=0, weights=weights2).reshape(1, -1)
        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)
    return sims







def infersent_sim(text_left, text_right):
    infersent  = load_infersent()
    raw_sentences1 = [sent1.raw for sent1 in text_left]
    raw_sentences2 = [sent2.raw for sent2 in text_right]

    infersent.build_vocab(raw_sentences1 + raw_sentences2, tokenize=True)
    embeddings1 = infersent.encode(raw_sentences1, tokenize=True)
    embeddings2 = infersent.encode(raw_sentences2, tokenize=True)

    infersent_sims = []
    for (emb1, emb2) in zip(embeddings1, embeddings2):
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        infersent_sims.append(sim)

    return infersent_sims


# def google_use_sim(sentences1, sentences2):
#     sts_input1 = tf.placeholder(tf.string, shape=(None))
#     sts_input2 = tf.placeholder(tf.string, shape=(None))

#     sts_encode1 = tf.nn.l2_normalize(embed(sts_input1))
#     sts_encode2 = tf.nn.l2_normalize(embed(sts_input2))
        
#     sim_scores = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         session.run(tf.tables_initializer())
      
#         [gse_sims] = session.run(
#             [sim_scores],
#             feed_dict={
#                 sts_input1: [sent1.raw for sent1 in sentences1],
#                 sts_input2: [sent2.raw for sent2 in sentences2]
#             })
#     return gse_sims

bench = [
    ("AVG-GLOVE", ft.partial(base_sim, embedding=glove, wosw=False)),
    ("AVG-GLOVE-WOSW", ft.partial(base_sim, embedding=glove, wosw=True)),
    ("AVG-GLOVE-TFIDF", ft.partial(base_sim, embedding=glove, wosw=False, tfidf=True)),
    ("AVG-GLOVE-TFIDF-WOSW", ft.partial(base_sim, embedding=glove, wosw=True, tfidf=True)),
    ("INF", infersent_sim)
]



def run_comparisons(df, benchmarks):
    sentences1 = [Text(s) for s in df['sent_left']]
    sentences2 = [Text(s) for s in df['sent_right']]

    pearson_cors, spearman_cors = [], []
    for label, method in benchmarks:
        sims = method(sentences1, sentences2)
        pearson_correlation = scipy.stats.pearsonr(sims, df['sim_score'])[0]
        pearson_cors.append(pearson_correlation)
        spearman_correlation = scipy.stats.spearmanr(sims, df['sim_score'])[0]
        spearman_cors.append(spearman_correlation)
        print(label, pearson_correlation, spearman_correlation)

    return pearson_cors, spearman_cors



pearson_results, spearman_results = {}, {}
pearson_results["STS-TRAIN"], spearman_results["STS-TRAIN"] = run_comparisons(sts_train, bench)
pearson_results["STS-DEV"], spearman_results["STS-DEV"] = run_comparisons(sts_dev, bench)

