#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################

import functools as ft
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

from utils import load_glove, load_google_use, load_infersent, load_sts_data

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
        self.tokens = [t for t in nltk.word_tokenize(self.raw)]
        self.tokens_wosw = [t for t in self.tokens if t not in STOP_WORDS]






def base_sim(text_left, text_right, embedding, lower=False, wosw=False, tfidf=False):
    """ compute sentence similairty through embeddings either w/wo:
        stopwords
        tf-idf frequencies
    """
    if tfidf:
        tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, lowercase=lower)
        tfidf_vectorizer = tfidf_vectorizer.fit([t.raw for t in text_left] + [t.raw for t in text_right])
        freq_dict = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))

    sims = []
    for (textl, textr) in zip(text_left, text_right):
        tokensl = textl.tokens_wosw if wosw else textl.tokens
        tokensr = textr.tokens_wosw if wosw else textr.tokens
        if lower:
            tokensl = [t.lower() for t in tokensl]
            tokensr = [t.lower() for t in tokensr]
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
    """ compute similarity through infersent embeddings
    """
    infersent  = load_infersent()
    raw_sentences1 = [sent.raw for sent in text_left]
    raw_sentences2 = [sent.raw for sent in text_right]

    infersent.build_vocab(raw_sentences1 + raw_sentences2, tokenize=True)
    embeddings1 = infersent.encode(raw_sentences1, tokenize=True)
    embeddings2 = infersent.encode(raw_sentences2, tokenize=True)

    infersent_sims = []
    for (emb1, emb2) in zip(embeddings1, embeddings2):
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        infersent_sims.append(sim)

    return infersent_sims


def google_use_sim(text_left, text_right):
    """ compute similarity through google use embeddings
    """
    guse = load_google_use()
    raw_sentences1 = [sent.raw for sent in text_left]
    raw_sentences2 = [sent.raw for sent in text_right]

    embeddings1 = guse(raw_sentences1)
    embeddings2 = guse(raw_sentences2)

    guse_sims = []
    for (emb1, emb2) in zip(embeddings1, embeddings2):
        sim = cosine_similarity(tf.reshape(emb1, (1, -1)), tf.reshape(emb2, (1, -1)))[0][0]
        guse_sims.append(sim)

    return guse_sims





bench = [
    ("AVG-GLOVE", ft.partial(base_sim, embedding=glove, wosw=False, lower=False)),
    ("AVG-GLOVE-LOW", ft.partial(base_sim, embedding=glove, wosw=False, lower=True)),
    ("AVG-GLOVE-WOSW", ft.partial(base_sim, embedding=glove, wosw=True, lower=False)),
    ("AVG-GLOVE-WOSW-LOW", ft.partial(base_sim, embedding=glove, wosw=True, lower=True)),
    ("AVG-GLOVE-TFIDF", ft.partial(base_sim, embedding=glove, wosw=False, tfidf=True)),
    ("AVG-GLOVE-TFIDF-LOW", ft.partial(base_sim, embedding=glove, wosw=False, lower=True, tfidf=True)),
    ("AVG-GLOVE-TFIDF-WOSW", ft.partial(base_sim, embedding=glove, wosw=True, tfidf=True)),
    ("AVG-GLOVE-TFIDF-WOSW-LOW", ft.partial(base_sim, embedding=glove, wosw=True, tfidf=True, lower=True)),
    ("INFERSENT", infersent_sim),
    ("USE", google_use_sim)
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




pearson_results_df = pd.DataFrame(pearson_results)
pearson_results_df = pearson_results_df.transpose()
pearson_results_df = pearson_results_df.rename(columns={i:b[0] for i, b in enumerate(bench)})
pearson_results_df[[b[0] for b in bench if b[0].startswith("AVG")]].plot(kind="bar").legend(loc="lower left")





plt.show()
