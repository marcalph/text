#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################
"""
text utils
"""
import csv
import logging
import tarfile
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tfhub
import torch
import wget
from annoy import AnnoyIndex

from src.encoders import InferSent

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)



################################
# data utils
################################
def load_data(path="data/t4.csv", size=5000):
    text_df = pd.read_csv(path).text.sample(size)
    return text_df.values.tolist()


def load_ner_data(path="data/entity-annotated-corpus/ner_dataset.csv", *args, **kwargs):
    df = pd.read_csv(path, engine="python", *args, **kwargs)
    df.columns = df.columns.map(lambda x:x[:4].lower())
    df.sent = df.sent.fillna(method="ffill")
    return df


def load_sts_data(file_path: str="data/stsbenchmark/sts-train.csv"):
    """ loads a subset of the STS dataset into a data frame
    """
    sent_pairs = []
    with tf.io.gfile.GFile(file_path, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pd.DataFrame(sent_pairs, columns=["sent_left", "sent_right", "sim_score"])


def get_sts_data():
    """ collect and extract sts data
    """
    sts_dataset = tf.keras.utils.get_file(
        fname="stsbenchmark.tar.gz",
        origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
        archive_format="tar",
        extract=True)
    archive = tarfile.open(sts_dataset)
    archive.extractall("data/")
    return None



def get_top_20k():
    """ downoad top 20k common words in english through wget
    """
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
    wget.download(url, out="data/freqs")
    return None


######################################
# pretrained encoder utils
######################################
def get_infersent():
    """ download infersent weight state pkl through wget
    """
    url = "https://dl.fbaipublicfiles.com/infersent/infersent1.pkl"
    wget.download(url, out="data/encoders/infersent")
    return None


def load_infersent(model_path: str="data/encoders/infersent/infersent1.pkl",
                   w2v_path: str="data/embeddings/glove.840B/glove.840B.300d.txt"):
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(model_path))
    infersent.set_w2v_path(w2v_path)
    return infersent


def load_google_use():
    """ download google use through tfhub
    """
    url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    guse = tfhub.load(url)
    return guse




######################################
# pretrained embeddings utils
######################################
def load_glove(glove_file: str="data/embeddings/glove.840B/glove.840B.300d.txt", prune: bool=False) -> Dict[str, np.ndarray]:
    """ load glove pretrained embeddings
        prune : keep top 20k lowercase word from glove vocab
        returns embedding as dict
    """
    glove = {}
    with open(glove_file, 'r', encoding="utf-8") as f:
        for line in f:
            splitted = line.split(" ")
            word = splitted[0]
            embedding = np.array([float(val) for val in splitted[1:]])
            glove[word] = embedding

    if prune:
        top20k = pd.read_csv("data/freqs/20k.txt", header=None)[0].values.tolist()
        to_prune = set(glove.keys())- set(top20k)
        for k in to_prune:
            glove.pop(k)

    return glove


def index_glove_embeddings(dict_embedding):
    """ index embeddings using annoy (spotify)
        returns:
            embedding_index an AnnoyIndex instance
            word_mapping dict of int to word in vocab
    """
    word_mapping = {i: word for i, word in enumerate(dict_embedding)}
    word_features = [dict_embedding[w] for _, w in word_mapping.items()]
    dim = next(iter(dict_embedding.values())).size
    logger.info("Building tree")
    embedding_index = AnnoyIndex(dim, metric="angular")
    for i, vec in enumerate(word_features):
        embedding_index.add_item(i, vec)
    embedding_index.build(25)
    logger.info("Tree built")
    return embedding_index, word_mapping



if __name__ == "__main__":
    glove = load_glove(prune=True)
    print(len(glove.keys()))
