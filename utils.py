import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score, roc_curve
import torch
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence


def get_label(data):
    label = np.zeros_like(data["label"], dtype=np.int32)
    label[data["label"] == "pants-fire"] = 1
    return label


def get_multi_label(data):
    label_dict = {
        "true": 5,
        "mostly-true": 4,
        "half-true": 3,
        "barely-true": 2,
        "false": 1,
        "pants-fire": 0
    }
    label = [label_dict[word_label] for word_label in data["label"].values]
    return np.array(label)


def plot_roc(y, probs, verbose=True):
    if len(probs.shape) > 1:
        preds = probs[:,1]
    else:
        preds = probs
    fpr, tpr, threshold = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)

    if verbose:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return roc_auc
    
    
def evaluate(model, x, y, verbose=True):
    probs = model.predict_proba(x)
    y_pred = probs[:, 1] > 0.5
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    if verbose:
        print("Accuracy: {:.2f}".format(acc * 100))
        print("F-1 score: {:.4f}".format(f1))
    return plot_roc(y, probs, verbose=verbose)
    
    
def instantiate(foo):
    return foo()
    

@instantiate
def Imputer():
    def impute(x_array):
        if type(x_array) == pd.core.series.Series:
            x_array = x_array.values
        return np.array(["unknown" if x is np.nan else x for x in x_array])
    return FunctionTransformer(impute)


@instantiate
def Vectorizer():
    return TfidfVectorizer(stop_words="english", min_df=0.0005, max_df=0.95, max_features=10_000)


@instantiate
def Splitter():
    def split_subject(x_array):
        if type(x_array) == pd.core.series.Series:
            x_array = x_array.values
        return np.array([x.replace(",", " ") for x in x_array])
    return FunctionTransformer(split_subject)


@instantiate
def Embedder():
    def embed(x_array):
        if type(x_array) == pd.core.series.Series:
            x_array = x_array.values
        glove_embedding = WordEmbeddings('glove')
        flair_embedding_forward = FlairEmbeddings('news-forward')
        flair_embedding_backward = FlairEmbeddings('news-backward')
        document_embeddings = DocumentPoolEmbeddings([flair_embedding_forward])
        embedding_list = []
        for text in tqdm(x_array):
            sentence = Sentence(text)
            document_embeddings.embed(sentence)
            embedding = sentence.get_embedding()
            embedding_list.append(embedding)
        embedding_list = torch.stack(embedding_list)
        return embedding_list.cpu().detach().numpy()
    return FunctionTransformer(embed)
