#!/usr/bin/env python3
# coding: utf-8
################################################
# authors                                      #
# marcalph	<marcalph@protonmail.com>			#
################################################
"""
embeddings
"""

import spacy
import tqdm
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from utils import load

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s \n %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)



# apply spacy  nlp pipeline
nlp = spacy.load("fr_core_news_md")
text_sample = load()[0]
doc = nlp(text_sample)


def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower]
    logger.info("querying done")
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    logger.info("sorting done")
    return by_similarity[:200]



def plot_embeddings(target_word):
    similar_words = most_similar(nlp.vocab[target_word])
    logger.debug("similar word querying done")
    words = [w.orth_ for w in similar_words]
    embeddings = [v.vector for v in similar_words]

    # tsne reduction
    mapped_embeddings = TSNE(n_components=2, perplexity=10).fit_transform(embeddings)
    logger.debug("projection done")
    df = pd.DataFrame({"x": me[:, 0], "y": me[:, 1], "label": w})
    p1 = sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="+", color="skyblue")
    for _, point in df.iterrows():
        p1.text(point['x'], point['y'], str(point['label']))

    plt.show()
    return words, mapped_embeddings

w, me = plot_embeddings("avocat")




# if __name__ == "__main__":
#     print(most_similar(nlp.vocab["chien"]))
#     logger.info(nlp.vocab["toujours"].cluster)



