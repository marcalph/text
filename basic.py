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
import annoy
from src.utils import load_google_use, load_sts_data, index_embeddings

# todo : add similarity based on annoy
# todo : add text eda based on encoding clustering

guse = load_google_use()

import plotly.figure_factory as ff

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
    # plot = go.Heatmap(
    #     showscale=False,
    #     z=corr,
    #     x=messages,
    #     y=messages)
    # fig = go.Figure(data=plot, layout=layout)
    corr_text = np.around(corr, decimals=2) # Only show rounded value (full value on hover)
    hover=[]
    for i in range(len(messages)):
        temp = []
        for j in range(len(messages)):
            temp.append(f"clause {i} and clause {j} have unscaled correlation of {corr_text[i][j]}")
        hover.append(temp)
    fig = ff.create_annotated_heatmap(corr, annotation_text=corr_text,
                                      text=hover,
                                      hoverinfo='text')

    return fig


@st.cache(allow_output_mutation=True)
def project_sent_embeddings(messages, embeddings):
    proj_embeddings = TSNE(n_components=2, perplexity=50, metric='cosine', init='pca').fit_transform(embeddings)
    x = proj_embeddings[:, 0]
    y = proj_embeddings[:, 1]
    plot = [go.Scatter(x=x,
                       y=y,
                       mode='markers',
                       text=messages,
                       textposition='bottom center',
                       hoverinfo='text',
                       marker=dict(size=5, opacity=.5, color="red"))]
    layout = go.Layout(title='Embedding Projector')
    fig = go.Figure(data=plot, layout=layout)
    return fig


@st.cache(hash_funcs={annoy.AnnoyIndex: lambda _: None})
def find_most_similar_texts(encoded_text, feature_index, text_map, top_n=20):
    """ return dict of word similarity
    """
    distances = feature_index.get_nns_by_vector(encoded_text, top_n, include_distances=True)
    resdict = {text_map[a]: distances[1][i] for i, a in enumerate(distances[0])}
    return resdict


# sentence similarity through correlation
# df = load_sts_data()
heatmap_messages = [
    "    This Contract negotiated in good faith between the Parties sets out the terms and conditions under which the Purchaser or Enabled Suppliers may issue Orders to the Supplier, and the Supplier shall execute such Orders, with respect to the manufacture and supply of the Products as defined in Article 4 below.",
    "    40 Notwithstanding any co-operation provided by the Purchaser, the Supplier shall be fully responsible for, if applicable:"
    "A. the detailed design of the Product in accordance with the requirements of the latest issue of the Specification;"
    "B. the development of the detailed component specifications within the Specification; and"
    "C. all changes to the design of the Product and the Specification.",
    "Unless otherwise agreed between the Parties, the Supplier undertakes to comply with the provisions included in the said Applicable Documents and their updates, and to apply these Applicable Documents at their latest issue.",
    "     Compliance with the Corrective Action Plan:"
    "The Supplier shall comply with the Corrective Action Plan."
    "In the event the Supplier fails to comply with the Corrective Action Plan, the Purchaser may give the Supplier written notice of such failure at any time thereafter. The Supplier shall remedy such failure within sixty (60) Days from the date of such notice.",
    "On request from the Purchaser, the Supplier shall demonstrate the means implemented in order to ensure its compliance with anti-corruption laws and regulations, and the Purchaser reserves the right to audit the Supplier in accordance with Article 6.2 “Audit and Inspection” in order to check the compliance of the Supplier to the applicable anti-corruption laws and regulations.",
    "The Supplier shall promptly provide an identification number to the Purchaser, either the FSCM (Federal Supply Code for Manufacturers) of the manufacturing plant, or its NSCM (NATO Supply Code for Manufacturers), or its CAGE (Commercial and Government Entity Code). Such code shall be communicated to the Purchaser as soon as available, and shall be quoted on all correspondence, product related documentation, certificates, manuals, etc.. Should the Supplier subcontract work to third parties, it shall ensure that such subcontractors comply with the above requirements.",
    "The Supplier shall ensure that each Product is packed and labelled in accordance with Applicable Laws. Where the Parties agree that reusable packaging and/or storage devices are not suitable for any Product or deliverable, the Supplier shall ensure that all non-reusable packaging and/or storage devices can be recycled in accordance with European environmental regulation EN13427.",
    "The Purchaser or Enabled Supplier reserves the right to accept or reject any Product delivered in advance of the Delivery Schedule and/or in excess of the quantity specified in the Order. If accepted, the payment for such early Delivery or excess quantity will be due according to the [Delivery Schedule initially agreed between the Parties.] [Delete and add the following alternative wording when Contract is subject to French law][provisions of Article 8.4 “Payment” hereof.]",
    "If an Excusable Delay occurs that causes or may cause a delay in the performance by a Party of its obligations under the Contract and/or an Order, such Party shall notify the other Party in writing of such Excusable Delay immediately after becoming aware of the same which notice shall:"
    "A. describe the event causing the Excusable Delay in reasonable detail; "
    "B. provide an evaluation of the obligations affected; "
    "C. indicate the probable extent of such delay; "
    "D. upon cessation of the event causing the Excusable Delay notify the other Party in writing of such cessation; and"
    "E. as soon as practicable after the removal of the cause of the delay, such Party shall resume the performance of its obligations under the Contract unless otherwise agreed.",
    "subject to the collaborative review set out in Article 7.2.2 above, liquidated damages may be set off against any payment outstanding or due to the Supplier.",
    "Should the Supplier’s quality system or the Product quality degrade, the Purchaser may perform a new qualification, all consequential costs being for the Supplier’s account. ",
    "Either Party may submit proposals to the other for modifications to the Product.",

    "For as long as a minimum of five (5) Aerospace Related Products on which the Product is incorporated are in regular operation, the Supplier undertakes that:"
    "A. It shall at its expense acknowledge that the Tools are of a standard and shall maintain them to such standard so as to meet the requirements of both the Purchaser and the Customer; and"
    "B. the Tools shall be free and clear of all liens, charges, mortgages or encumbrances and rights of others of any kind whatsoever, and the Supplier shall fully indemnify and hold the Purchaser harmless in this regard. "
    "C. The Supplier shall, if required, declare in due time, the Tooling to the relevant tax authorities and pay any corresponding tax.",
    "Shall be free and clear of all liens, charges, mortgages or encumbrances and rights of others of any kind whatsoever and the Supplier shall fully indemnify and hold the Purchaser or Enabled Supplier harmless in this regard",
    "f a notice of termination of this Contract and/or the relevant Orders is served, it shall specify the effective date of termination. At the effective date of termination, the Purchaser shall have the rights set out in Article 15.5 “Termination Procedures” and the Parties shall proceed to a termination account accordingly.",
    "The Supplier shall, when carrying out work of any kind in the premises of the Purchaser or in such other premises as the Order so directs, effect and maintain a General Third Party Liability Insurance for an amount satisfactory to the Purchaser and in any event not less than [value in words] Euros / US Dollars (€ / $[value in numbers]) per occurrence.",
    "In consideration of the licence received under Article 17.2.3.2 below, the Supplier hereby grants to the Purchaser, for the duration of the relevant Supplier’s Foreground IP Rights, a free of charge, irrevocable, world-wide licence to use and have used any of the Supplier’s Foreground IP for the purpose of ensuring continuity of supply of the Product or in connection with the Aerospace Related Product. Such licence shall include the right to sub-licence such Supplier’s Foreground IP Rights to third parties free of charge.",
    "The terms of the Non-Disclosure Agreement ('NDA') incorporated herein via Annex E “Non-Disclosure Agreement” shall apply to any and all information, of any nature whatsoever, exchanged under or in connection with this Contract. As an exception to the terms of the NDA, the Parties hereby agree that the Purchaser is entitled to disclose to the Enabled Suppliers any information under this Contract which is necessary to allow the implementation of Article 4.3 “Conditions Applicable to Enabled Suppliers”.",
    "This Contract, including this Article 21, shall not be amended except by specific agreement in writing signed by the duly authorised representatives of the Parties, and recorded in Annex M “Table of Content Evolution”.",
    "Where, however, the provisions of any such Applicable Law may be waived, they are hereby waived by the Parties to the fullest extent permitted by such law, with the result that the Contract shall be valid and binding and enforceable in accordance with its terms. "
]
# heatmap_messages = df.sent_left.sample(50, random_state=42).values.tolist()
heatmap_embeddings = guse(heatmap_messages)
heat_plot = plot_similarity_heatmap(heatmap_messages, heatmap_embeddings)


# sentence similarity through projection
# projection_messages = df.sent_left[:2000]
# projection_embeddings = guse(projection_messages).numpy()
# proj_plot = project_sent_embeddings(projection_messages, projection_embeddings)


# embedding_dict = dict(zip(projection_messages, projection_embeddings))
# feature_index, text_map = index_embeddings(embedding_dict)


# demo
st.write("# similarity")


parts = ["heatmap", "projection", "similarity"]
part = st.sidebar.selectbox(
    'Which part do you wish to display ?',
    parts)


if part == parts[0]:
    st.markdown("## similarity heatmap")
    st.write(heatmap_messages)
    st.write(heat_plot)
elif part == parts[1]:
    st.markdown("## dimensionality reduction")
    st.write(proj_plot)
elif part == parts[2]:
    st.markdown("## similar texts")
    text_input = st.text_input("text", "the cat is rolling over")
    encoded_text = guse([text_input]).numpy()
    similar = find_most_similar_texts(encoded_text[0], feature_index, text_map)
    st.write(list(similar.keys()))
