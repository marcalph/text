#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################
""" simple semantic similarity demo
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import TSNE
from src.utils import load_google_use, load_sts_data

# todo : add similarity based on annoy
# todo : change example dataset



guse = load_google_use()

@st.cache(allow_output_mutation=True)
def plot_similarity_heatmap(messages, embeddings):
    corr = np.inner(embeddings, embeddings)
    layout = go.Layout(
        xaxis=dict(
            autorange=True,
            showgrid=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            ticks='',
            showticklabels=False
        )
    )
    plot = go.Heatmap(
                    showscale=False,
                    z=corr,
                    x=messages,
                    y=messages)
    fig = go.Figure(data=plot, layout=layout)
    return fig



@st.cache(allow_output_mutation=True)
def project_sent_embeddings(messages, embeddings):
    proj_embeddings = TSNE(n_components=2, perplexity=50, metric='cosine', init='pca').fit_transform(embeddings)
    x = proj_embeddings[:,0]
    y = proj_embeddings[:,1]
    # df = pd.DataFrame({"x":x, "y":y, "z":z, "words":selected_words})
    plot = [go.Scatter(x = x,
                y = y,
                mode = 'markers',
                text = messages,
                textposition='bottom center',
                hoverinfo = 'text',
                marker=dict(size=5,opacity=.5, color="red"))]
    layout = go.Layout(title='Embedding Projector')
    fig = go.Figure(data=plot, layout=layout)
    return fig


# sentence similarity through correlation
df = load_sts_data()
heatmap_messages = df.sent_left.sample(50, random_state=42).values.tolist()
heatmap_embeddings = guse(heatmap_messages)
heat_plot = plot_similarity_heatmap(heatmap_messages, heatmap_embeddings)


#sentence similarity through projection
projection_messages = df.sent_left[:5000]
projection_embeddings = guse(projection_messages)
proj_plot = project_sent_embeddings(projection_messages, projection_embeddings)




# demo
st.write("# similarity")


parts = ["heatmap", "projection"]


part = st.sidebar.selectbox(
    'Which part do you wish to display ?',
    parts)


if part == parts[0]:
    st.write(heatmap_messages)
    st.write(heat_plot)
elif part == parts[1]:
    st.write(proj_plot)

