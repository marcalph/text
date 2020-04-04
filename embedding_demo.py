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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearn.manifold import TSNE
from wordcloud import WordCloud

from src.utils import index_glove_embeddings, load_glove, search_index

matplotlib.use('Agg')


#todo update visualisation and fix caching

# code
@st.cache(allow_output_mutation=True)
def load_and_index():
    glove = load_glove("data/embeddings/glove.6B/glove.6B.50d.txt")
    index, mapping = index_glove_embeddings(glove)
    return glove, index, mapping


@st.cache(hash_funcs={annoy.AnnoyIndex: lambda _:None})
def find_most_similar(word, top_n=30):
    return search_index(glove[word], index, mapping, top_n)


# demo
st.write("# embeddings")

glove, index, mapping = load_and_index()

st.write("## basics")
# show example of embedding
if st.checkbox("display embedding"):
    st.write(glove["cat"])


# show most similar words to query
word = st.text_input("word query", "cat" )
similar = find_most_similar(word)
st.write(similar.keys())

if st.checkbox("display visulisation"):
    wordcloud = WordCloud(max_font_size=100, relative_scaling=0.8, background_color="white", colormap="YlOrRd_r")\
        .generate_from_frequencies(similar)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()


# basic operations
st.markdown("## operations")
pos_words = st.text_input("positive words", "king, woman").split(", ")
neg_words = st.text_input("negative words", "man").split(", ")
st.latex("+".join(pos_words)+"-"+"-".join(neg_words))

request = 0
for w in pos_words:
    request += glove[w]
for w in neg_words:
    request -= glove[w]

st.write(list(search_index(request, index, mapping).keys())[1:])


# visualisation
st.markdown("## visualisation")
query = st.text_input("projection query", "france")



@st.cache(hash_funcs={annoy.AnnoyIndex: lambda _:None})
def compute_viz(word_query):
    selected_words = list(find_most_similar(word_query, 100).keys())
    selected_embeddings = np.vstack([glove[w] for w in selected_words])
    mapped_embeddings = TSNE(n_components=3, metric='cosine', init='pca').fit_transform(selected_embeddings)
    x = mapped_embeddings[:,0]
    y = mapped_embeddings[:,1]
    z = mapped_embeddings[:,2]
    # df = pd.DataFrame({"x":x, "y":y, "z":z, "words":selected_words})
    plot = [go.Scatter3d(x = x,
                    y = y,
                    z = z,
                    mode = 'markers+text',
                    text = selected_words,
                    textposition='bottom center',
                    hoverinfo = 'text',
                    marker=dict(size=5,opacity=0.8))]
    layout = go.Layout(title='Ok Computer lyrics')
    fig = go.Figure(data=plot, layout=layout)
#     fig = px.scatter_3d(df, x='x', y='y', z='z',
#                     mode = 'markers+text',
#                     text = "words",
#                     textposition='bottom center',
#                     hoverinfo = 'text',
#                     marker=dict(size=5,opacity=0.8) )
    return fig

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
