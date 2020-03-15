import numpy as np
import logging
from annoy import AnnoyIndex


import streamlit as st




logger = logging.getLogger(__file__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)


@st.cache
def load_glove(glove_file):
    glove = {}
    with open(glove_file, 'r', encoding="utf-8") as f:
        for line in f:
            splitted = line.split()
            word = splitted[0]
            embedding = np.array([float(val) for val in splitted[1:]])
            glove[word] = embedding
    return glove

def index_glove_embeddings(glove):
    word_mapping = {i: word for i, word in enumerate(glove)}
    word_features = [glove[w] for i,w in word_mapping.items()]
    logging.info("Building tree")
    
    embedding_index = AnnoyIndex(50, metric="angular")
    for i, vec in enumerate(word_features):
        embedding_index.add_item(i, vec)
    embedding_index.build(20)
    # word_index = index_features(word_features, n_trees=20, dims=300)
    logging.info("Tree built")
    return embedding_index, word_mapping



def search_index(value, index, mapping, top_n=10):
    distances = index.get_nns_by_vector(value, top_n, include_distances=True)
    return [[a, mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


# main
def main():
    glove = load_glove("C:/Users/marc.alphonsus/Desktop/projects/personnal/tools/glove.6B.50d.txt")
    index, mapping = index_glove_embeddings(glove)
    st.write(search_index(glove['cat'], index, mapping, topn))


st.write("# test demo")
topn = st.slider("topn")
main()