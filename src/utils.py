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
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s", "%Y-%m-%d %H:%M")
ch.setFormatter(formatter)
logger.addHandler(ch)

def load_data(path="data/t4.csv", size=5000):
    text_df = pd.read_csv(path).text.sample(size)
    return text_df.values.tolist()



def load_ner_data(path="data/entity-annotated-corpus/ner_dataset.csv", *args, **kwargs):
    df = pd.read_csv(path, engine="python", *args, **kwargs)
    df.columns = df.columns.map(lambda x:x[:4].lower())
    df.sent = df.sent.fillna(method="ffill")
    return df


if __name__ == "__main__":
    logger.info("load data")
    df = load_ner_data()
    logger.info(df.head(30))



