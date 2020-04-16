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
    also provides a tensorflow projector like viz
"""
import annoy
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearn.manifold import TSNE
from wordcloud import WordCloud

from src.utils import index_glove_embeddings, load_glove

matplotlib.use('Agg')


#todo update visualisation and fix caching

# code
@st.cache(allow_output_mutation=True)
def load_and_index():
    glove = load_glove("data/embeddings/glove.840B/glove.840B.300d.txt", prune=True)
    index, mapping = index_glove_embeddings(glove)
    return glove, index, mapping

glove, index, mapping = load_and_index()

@st.cache(hash_funcs={annoy.AnnoyIndex: lambda _:None})
def find_most_similar_to_word(word, glove=glove, index=index, mapping=mapping, top_n=20, drop_query=True):
    """ return dict of word similarity
    """
    distances = index.get_nns_by_vector(glove[word], top_n, include_distances=True)
    resdict = {mapping[a] : 1/(distances[1][i]+0.1) for i, a in enumerate(distances[0])}
    if drop_query:
        resdict.pop(word, None)
    return resdict

@st.cache(hash_funcs={annoy.AnnoyIndex: lambda _:None})
def find_most_similar_to_value(value, index=index, mapping=mapping, top_n=20):
    distances = index.get_nns_by_vector(value, top_n, include_distances=True)
    resdict = {mapping[a] : 1/(distances[1][i]+0.1) for i, a in enumerate(distances[0])}
    return resdict


@st.cache(hash_funcs={annoy.AnnoyIndex: lambda _:None})
def compute_viz(word_query):
    selected_words = list(find_most_similar_to_word(word_query, top_n=15, drop_query=False).keys())
    selected_embeddings = np.vstack([glove[w] for w in selected_words]+
                                    [glove[w] for w in np.random.choice(list(glove.keys()), 1000, replace=False)])
    print("len(selected_embeddings)")
    print(len(selected_embeddings))
    tags = selected_words + [""] * (len(selected_embeddings) - len(selected_words))
    colors = ["orangered" if x!="" else "steelblue" for x in tags]
    mapped_embeddings = TSNE(n_components=3, metric='cosine', init='pca').fit_transform(selected_embeddings)
    x = mapped_embeddings[:,0]
    y = mapped_embeddings[:,1]
    z = mapped_embeddings[:,2]
    # df = pd.DataFrame({"x":x, "y":y, "z":z, "words":selected_words})
    plot = [go.Scatter3d(x = x,
                    y = y,
                    z = z,
                    mode = 'markers+text',
                    text = tags,
                    textposition='bottom center',
                    hoverinfo = 'text',
                    marker=dict(size=5,opacity=.5, color=colors))]
    layout = go.Layout(title='Embedding Projector')
    fig = go.Figure(data=plot, layout=layout)
    return fig




# demo
st.write("# embeddings")

part = st.sidebar.selectbox(
    'Which part do you wish to display ?',
    range(3))

if part == 0:
    st.write("## basics")
    # show example of embedding
    if st.checkbox("show embedding vector"):
        st.write(glove["cat"])

    # show most similar words to query
    word = st.text_input("word", "cat" )

    similar = find_most_similar_to_word(word)
    worddict = similar
    worddict[word]=5
    st.write(similar.keys())
    wordcloud = WordCloud(max_font_size=100, relative_scaling=0.8, background_color="white", colormap="Greens")\
            .generate_from_frequencies(worddict)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

elif part == 1:
    # basic operations
    st.markdown("## operations")
    pos_words = st.text_input("positive words", "king, woman").translate(str.maketrans('', '', string.punctuation)).split()
    neg_words = st.text_input("negative words", "man").translate(str.maketrans('', '', string.punctuation)).split()

    st.latex("+".join(pos_words)+"-"+"-".join(neg_words))

    request = 0
    for w in pos_words:
        request += glove[w]
    for w in neg_words:
        request -= glove[w]
    ops_dict = find_most_similar_to_value(request, index, mapping)
    for w in pos_words+neg_words:
        ops_dict.pop(w, None)
    st.write(ops_dict.keys())

elif part == 2:
    # visualisation
    st.markdown("## visualisation")
    query = st.text_input("projection query", "france")
    fig = compute_viz(query)
    st.write(fig)









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
