import pickle
import enum
import numpy as np
import torch.nn as nn
import torch
import transformers

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from BERT_Arch import BERT_Arch
from roBERTa_Arch import roBERTa_Arch
from DistilBERT_Arch import DistilBERT_Arch
from transformers import AutoModel, BertTokenizerFast, RobertaTokenizerFast, DistilBertModel, DistilBertTokenizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


label = {
    0: 'left',
    1: 'right'
}

device = torch.device('cpu')

# open models

# distilbert 
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBERT_Arch(distilbert)
distilbert_model.load_state_dict(torch.load('distilBERT_weights.pt', map_location=device))
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', return_dict=False)

# pretrained bert
bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
bert_model = BERT_Arch(bert)
bert_model.load_state_dict(torch.load('bert_weights3.0.pt', map_location=device))
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', return_dict=False)

# roberta model
roberta = AutoModel.from_pretrained('roberta-base', return_dict=False)
roberta_model = roBERTa_Arch(roberta)
roberta_model.load_state_dict(torch.load('roberta_weights.pt', map_location=device))
roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', return_dict=False)

# SVM
clf_SVM = pickle.load(open('linear_model.pkl', 'rb'))


def predict_BERT(text):
    text_encoded = bert_tokenizer.batch_encode_plus(
        [text],
        max_length = 512,
        padding='longest',
        truncation=True
    )

    t_seq = torch.tensor(text_encoded['input_ids'])
    t_mask = torch.tensor(text_encoded['attention_mask'])

    pred = bert_model(t_seq, t_mask).detach().cpu().numpy()
    return label[np.argmax(pred, axis=1)[0]]


def predict_roBERTa(text):
    text_encoded = roberta_tokenizer.batch_encode_plus(
        [text],
        max_length = 512,
        padding='longest',
        truncation=True
    )


    t_seq = torch.tensor(text_encoded['input_ids'])
    t_mask = torch.tensor(text_encoded['attention_mask'])

    pred = roberta_model(t_seq, t_mask).detach().cpu().numpy()
    return label[np.argmax(pred, axis=1)[0]]


def predict_SVM(text):
    pred = clf_SVM.predict([text])[0]
    return label[pred]

def predict_distilBERT(text):
    text_encoded = distilbert_tokenizer.batch_encode_plus(
        [text],
        max_length = 512,
        padding='longest',
        truncation=True
    )

    t_seq = torch.tensor(text_encoded['input_ids'])
    t_mask = torch.tensor(text_encoded['attention_mask'])

    pred = distilbert_model(t_seq, t_mask).detach().cpu().numpy()
    return label[np.argmax(pred, axis=1)[0]]
