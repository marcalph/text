#!/usr/bin/env python3
# coding: utf-8
################################################
# authors                                      #
# marcalph	<marcalph@protonmail.com>			#
################################################
"""
text loading utils
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s %(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)

def load(path="data/t4.csv", size=5000):
    text_df = pd.read_csv(path).text.sample(size)
    return text_df.values.tolist()


def subset(sample, keyword):
    return [text for text in sample if keyword.lower in text.lower()]


if __name__ == "__main__":
    text_sample = load()
    logger.info(text_sample[:10])



