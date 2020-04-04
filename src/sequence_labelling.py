#!/usr/bin/env python3
# coding: utf-8
########################################
# authors                              #
# marcalph https://github.com/marcalph #
########################################
""" sequence labelling through a simple pytorch ner model
"""

import math
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import BucketIterator, Dataset, Example, Field
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from utils import load_ner_data

# todo start
# improve loading
# merge spacy ner sequence labelling script with sequence labelling
# todo end

text_field = Field(sequential=True, tokenize=lambda x:x, include_lengths=True) # default tokenize is str.split
tag_field = Field(sequential=True, tokenize=lambda x:x, is_target=True)


def make_examples(df:pd.DataFrame):
    examples = []
    fields = {'tag': ('tag', tag_field),
              'word': ('word', text_field)}

    for _,row in tqdm(df.groupby(["sent"]).agg({"word":list, "tag":list}).iterrows()):
        # for row in sent:
        example = Example.fromdict(row, fields)
        examples.append(example)

    return Dataset(examples, fields=[('tag', tag_field), ('word', text_field)]).split()


df = load_ner_data()
print(df.head())
train_torchds, val_torchds = make_examples(df)
print(train_torchds.fields)
print(train_torchds[0].__dict__.keys())
print(train_torchds[0].word)
print(train_torchds[0].tag)


vocab_size = 20000
text_field.build_vocab(train_torchds, max_size=vocab_size)
tag_field.build_vocab(train_torchds)



train_iter, val_iter = BucketIterator.splits((train_torchds, val_torchds),
                                            batch_sizes=(64,64),
                                            sort_within_batch=True, 
                                            repeat=False,
                                            sort_key=lambda x: len(x.word))


class LSTMbaseline(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, embeddings=None):
        super().__init__()
        #todo add pretrained embeddings
        if embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embeddings = nn.Embedding.from_pretrained(embeddings)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=1)
        self.dropout_layer = nn.Dropout(p=0.5)
        self.hidden2tag = nn.Linear(2*hidden_dim, output_size)

    def forward(self, batch_text, batch_lengths):
        embeddings = self.embeddings(batch_text)
        packed_seqs = pack_padded_sequence(embeddings, batch_lengths)
        lstm_output, _ = self.lstm(packed_seqs)
        lstm_output, _ = pad_packed_sequence(lstm_output)
        lstm_output = self.dropout_layer(lstm_output)
        logits = self.hidden2tag(lstm_output)
        return logits



def remove_predictions_for_masked_items(predicted_labels, correct_labels):
    predicted_labels_without_mask = []
    correct_labels_without_mask = []
    for p, c in zip(predicted_labels, correct_labels):
        if c > 1:
            predicted_labels_without_mask.append(p)
            correct_labels_without_mask.append(c)
    return predicted_labels_without_mask, correct_labels_without_mask



def train(model, train_iter, val_iter, batch_size, max_epochs, num_batches, patience, output_path):
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # mask <pad> labels
    optimizer = optim.Adam(model.parameters())

    train_f_score_history = []
    val_f_score_history = []
    no_improvement = 0
    for epoch in range(max_epochs):
        total_loss = 0
        predictions, correct = [], []
        for batch in tqdm(train_iter, total=num_batches, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            text_length, cur_batch_size = batch.word[0].shape
            pred = model(batch.word[0].to(device), batch.word[1].to(device)).view(cur_batch_size*text_length, NUM_CLASSES)
            gold = batch.tag.to(device).view(cur_batch_size*text_length)
            loss = criterion(pred, gold)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, pred_indices = torch.max(pred, 1)
            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(batch.tag.view(cur_batch_size*text_length).numpy())
            predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels, correct_labels)
            predictions += predicted_labels
            correct += correct_labels

        train_scores = precision_recall_fscore_support(correct, predictions, average="micro")
        train_f_score_history.append(train_scores[2])

        print("Total training loss:", total_loss)
        print("Training performance:", train_scores)

        total_loss = 0
        predictions, correct = [], []
        for batch in val_iter:
            text_length, cur_batch_size = batch.word[0].shape
            pred = model(batch.word[0].to(device), batch.word[1].to(device)).view(cur_batch_size * text_length, NUM_CLASSES)
            gold = batch.tag.to(device).view(cur_batch_size * text_length)
            loss = criterion(pred, gold)
            total_loss += loss.item()

            _, pred_indices = torch.max(pred, 1)
            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(batch.tag.view(cur_batch_size*text_length).numpy())
            predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels, correct_labels)
            predictions += predicted_labels
            correct += correct_labels
        val_scores = precision_recall_fscore_support(correct, predictions, average="micro")
        print("Total development loss:", total_loss)
        print("Development performance:", val_scores)
        val_f = val_scores[2]
        if len(val_f_score_history) > patience and val_f < max(val_f_score_history):
            no_improvement += 1
        elif len(val_f_score_history) == 0 or val_f > max(val_f_score_history):
            print("Saving model.")
            torch.save(model, output_path)
            no_improvement = 0

        if no_improvement > patience:
            print("Development F-score does not improve anymore. Stop training.")
            val_f_score_history.append(val_f)
            break

        val_f_score_history.append(val_f)
    return train_f_score_history, val_f_score_history



device="cpu"

BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_CLASSES = len(tag_field.vocab)
MAX_EPOCHS = 50
PATIENCE = 3
OUTPUT_PATH = "./baseline"
num_batches = math.ceil(len(train_torchds) / BATCH_SIZE)


tagger = LSTMbaseline(EMBEDDING_DIM, HIDDEN_DIM, vocab_size+2, NUM_CLASSES)
train_fscore, val_fscore = train(tagger.to(device), train_iter, val_iter, BATCH_SIZE, MAX_EPOCHS,
                       num_batches, PATIENCE, OUTPUT_PATH)


# plot train/val score
df = pd.DataFrame({'epochs': range(0,len(train_fscore)), 
                  'train_f': train_fscore, 
                   'val_f': val_fscore})
plt.plot('epochs', 'train_f', data=df, color='blue', linewidth=2)
plt.plot('epochs', 'val_f', data=df, color='green', linewidth=2)
plt.legend()
plt.show()


tagger = torch.load(OUTPUT_PATH)
tagger.eval()


def test(model, test_iter, batch_size, labels, target_names): 
    total_loss = 0
    predictions, correct = [], []
    for batch in test_iter:

        text_length, cur_batch_size = batch.word[0].shape

        pred = model(batch.word[0].to(device), batch.word[1].to(device)).view(cur_batch_size * text_length, NUM_CLASSES)
        gold = batch.tag.to(device).view(cur_batch_size * text_length)

        _, pred_indices = torch.max(pred, 1)
        predicted_labels = list(pred_indices.cpu().numpy())
        correct_labels = list(batch.tag.view(cur_batch_size*text_length).numpy())

        predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels, 
                                                                               correct_labels)

        predictions += predicted_labels
        correct += correct_labels
    print(classification_report(correct, predictions, labels=labels, target_names=target_names))



labels = tag_field.vocab.itos[3:]
labels = sorted(labels, key=lambda x: x.split("-")[-1])
label_idxs = [tag_field.vocab.stoi[l] for l in labels]

test(tagger, val_iter, BATCH_SIZE, labels = label_idxs, target_names = labels)
