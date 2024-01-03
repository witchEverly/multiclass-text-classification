import pandas as pd
import numpy as np

import string
import re

import nltk
from nltk.stem.porter import *

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


def tokenize(text) -> list:
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.
    """
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]
    tokenized_words = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokenized_words


def create_stemwords(words) -> list:
    """
    Given a list of tokens/words, returns a new list with each word
    stemmed using a PorterStemmer.
    """
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in words]
    return stemmed_words


def naive_bayes_pipeline(X_train: pd.Series,
                         y_train: pd.Series,
                         X_test: pd.Series,
                         y_test: pd.Series,
                         sampling_method: str = None,
                         **kwargs) -> tuple:
    """
    Create a TF-IDF vectorized Naive Bayes model pipeline and fit it to the training data.

    Parameters:
    X_train, y_train, X_test, y_test: Training and testing data and labels.
    sampling_method: The oversampling method to be used ('smote', 'adasyn', 'ros').
    **kwargs: Additional keyword arguments for TfidfVectorizer and MultinomialNB.

    Returns:
    Tuple: A tuple containing the trained pipeline, predictions, and classification report.
    """
    sampling_methods = {'smote': SMOTE,
                        'adasyn': ADASYN,
                        'ros': RandomOverSampler}

    sampler = sampling_methods.get(sampling_method)

    if sampler:
        pipeline = ImbPipeline([
            ('tfidf', TfidfVectorizer(**kwargs.get('tfidf_params', {}))),
            ('sampler', sampler()),
            ('nb', MultinomialNB(**kwargs.get('nb_params', {})))
        ])
    else:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**kwargs.get('tfidf_params', {}))),
            ('nb', MultinomialNB(**kwargs.get('nb_params', {})))
        ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return pipeline, predictions

    # EXAMPLE USAGE:
    # pipeline, predictions = naive_bayes_pipeline(...)


def logistic_regression_pipeline(X_train: pd.Series,
                                    y_train: pd.Series,
                                    X_test: pd.Series,
                                    y_test: pd.Series,
                                    sampling_method: str = None,
                                    **kwargs) -> tuple:
    """
    Create a TF-IDF vectorized Logistic Regression model pipeline and fit it to the training data. Allows passing custom parameters to TfidfVectorizer, sampling method, and LogisticRegression.

    Parameters:
    X_train, y_train, X_test, y_test: Training and testing data and labels.
    sampling_method: The oversampling method to be used ('smote', 'adasyn', 'ros').
    **kwargs: Additional keyword arguments for TfidfVectorizer and LogisticRegression.

    Returns:
    Tuple: A tuple containing the trained pipeline, predictions, and classification report.
    """



    sampling_methods = {'smote': SMOTE,
                        'adasyn': ADASYN,
                        'ros': RandomOverSampler}

    sampler = sampling_methods.get(sampling_method)

    if sampler:
        pipeline = ImbPipeline([
            ('tfidf', TfidfVectorizer(**kwargs.get('tfidf_params', {}))),
            ('sampler', sampler()),
            ('logreg', LogisticRegression(**kwargs.get('logreg_params', {})))
        ])
    else:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**kwargs.get('tfidf_params', {}))),
            ('logreg', LogisticRegression(**kwargs.get('logreg_params', {})))]
        )

    scale = StandardScaler(with_mean=False)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**kwargs.get('tfidf_params', {}))),
        ('scale', scale),
        ('logreg', LogisticRegression(**kwargs.get('logreg_params', {})))]
    )
    pipeline.fit(X_train, y_train)

    pipeline.fit_transform(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return pipeline, predictions

    # EXAMPLE USAGE:
    # pipeline, predictions = logistic_pipeline(...)