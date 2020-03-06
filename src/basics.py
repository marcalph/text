#!/usr/bin/env python3
# coding: utf-8
################################################
# authors                                      #
# marcalph	<marcalph@protonmail.com>			#
################################################
"""
embeddings
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import spacy
import tabulate

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import load
from src.utils import load

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s \n %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)



# apply spacy  nlp pipeline
nlp = spacy.load("fr_core_news_md")


def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower]
    logger.info("querying done")
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    logger.info("sorting done")
    return by_similarity[:100]


def plot_embeddings(similar_words):
    logger.debug("similar word querying done")
    words = [w.orth_ for w in similar_words]
    embeddings = [v.vector for v in similar_words]

    # tsne reduction
    mapped_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    logger.debug("projection done")
    df = pd.DataFrame({"x": mapped_embeddings[:, 0], "y": mapped_embeddings[:, 1], "label": words})
    p1 = sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="+", color="skyblue")
    for i, point in df.iterrows():
        p1.text(point['x'], point['y'], str(point['label']),  alpha=round(random.random()/5,2))
    plt.show()
    return words, mapped_embeddings




sims = most_similar(nlp.vocab["France"])


plot_embeddings(sims)

[t.text for t in sims]



sample = ("Aaron Swartz, né le 8 novembre 1986 à Chicago et mort le 11 janvier 2013 à New York, est un informaticien, écrivain, militant politique et hacktiviste américain."
          "Fervent partisan de la liberté numérique, il consacra sa vie à la défense de la « culture libre », convaincu que l'accès à la connaissance est un moyen d'émancipation et de justice.")


