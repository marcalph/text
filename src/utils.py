#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################
"""
text utils
"""
import logging
from typing import Dict
import pandas as pd
import numpy as np
from annoy import AnnoyIndex

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)


# data utils
def load_data(path="data/t4.csv", size=5000):
    text_df = pd.read_csv(path).text.sample(size)
    return text_df.values.tolist()


def load_ner_data(path="data/entity-annotated-corpus/ner_dataset.csv", *args, **kwargs):
    df = pd.read_csv(path, engine="python", *args, **kwargs)
    df.columns = df.columns.map(lambda x:x[:4].lower())
    df.sent = df.sent.fillna(method="ffill")
    return df


# pretrained embeddings utils
def load_glove(glove_file: str):# -> Dict[str, np.ndarray]:
    """ load glove pretrained embeddings
        returns embedding as dict
    """
    glove = {}
    with open(glove_file, 'r', encoding="utf-8") as f:
        for line in f:
            splitted = line.split()
            word = splitted[0]
            embedding = np.array([float(val) for val in splitted[1:]])
            glove[word] = embedding
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
    embedding_index.build(dim//3)
    logger.info("Tree built")
    return embedding_index, word_mapping


def search_index(value, index, mapping, top_n=10):
    distances = index.get_nns_by_vector(value, top_n, include_distances=True)
    logger.debug(distances)
    resdict = {mapping[a] : 1/(distances[1][i]+0.1) for i, a in enumerate(distances[0])}
    logger.debug(resdict)
    return resdict

if __name__ == "__main__":
    pass

