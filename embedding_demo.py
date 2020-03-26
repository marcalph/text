#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################
""" simple embedding demo
    useful to
    > show what an embedding is
    > give intuition about encoded info
"""
import logging
import time

import annoy
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from src.utils import index_glove_embeddings, load_glove, search_index

#todo fix logs and caching
logger = logging.getLogger(__file__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)




# code
@st.cache(allow_output_mutation=True)
def load_and_index():
    glove = load_glove("data/embeddings/glove.6B/glove.6B.50d.txt")
    index, mapping = index_glove_embeddings(glove)
    return glove, index, mapping


def find_most_similar(word, top_n=30):
    return search_index(glove[word], index, mapping, top_n)





# demo
st.write("# embedding demo")

glove, index, mapping = load_and_index()

# show example of embedding
if st.checkbox("Show embedding"):
    st.write(glove["cat"])

word = st.text_input("word query", "cat" )
similar = find_most_similar(word)

wordcloud = WordCloud(max_font_size=100, relative_scaling=0.8, background_color="white", colormap="YlOrRd_r")\
    .generate_from_frequencies(similar)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot()


# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(10):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'

# option = st.sidebar.selectbox(
#     'Which number do you like ?',
#      df['first column'])

# 'You selected:', option

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])

# if st.checkbox('Show dataframe'):
#     chart_data
# st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)
