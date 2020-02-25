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
from src import load


# apply spacy  nlp pipeline
text_sample = load()
doc = nlp(text_sample)