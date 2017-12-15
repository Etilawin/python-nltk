# https://www.youtube.com/watch?v=nla4C-VYNEU&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=15

# Classify a text (spam | not spam)

import nltk
import random
from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier # Wrapper to include the scikit-learn algorithm within the NLTK classifier itself
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Training
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior occurences x likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)

# classifier_f = open("naivebays.pickle", 'rb')
# classifier = pickle.load(classifier_f)
# classifier_f.close()

print("Original Naive Bayes Algo accuracy :", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB()) # Creates a classifier (converts it to a nltk classifier)
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent :", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# GaussianNB, BernoulliNB

# GaussianNB_classifier = SklearnClassifier(GaussianNB()) # Creates a classifier (converts it to a nltk classifier)
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy percent :", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB()) # Creates a classifier (converts it to a nltk classifier)
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent :", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression()) # Creates a classifier (converts it to a nltk classifier)
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent :", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier()) # Creates a classifier (converts it to a nltk classifier)
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent :", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC()) # Creates a classifier (converts it to a nltk classifier)
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent :", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC()) # Creates a classifier (converts it to a nltk classifier)
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent :", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC()) # Creates a classifier (converts it to a nltk classifier)
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent :", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
