# coding=UTF8
##################################################################
# Step 1: Lemmatize the data and save to avoid reruns.
# From: https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
# TODO: Remeber to ask permission to use if left in code.
#
##################################################################
import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize, regexp_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

token_re = r'\w+-\w+|\w+|[^\w\s]+'


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.digit      = set(string.digits)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        # TODO JPL: Data contains sentences without space: "First sent.Second sent."
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            # TODO JPL: Part of speech analysis obsolete for keywords (different meaning)
            #for token, tag in pos_tag(wordpunct_tokenize(sent)):
            for token, tag in pos_tag(regexp_tokenize(sent, token_re, gaps=False)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                #token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation or plain number, ignore token and continue
                if all(char in self.punct or char in self.digit for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)