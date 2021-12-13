# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:58:12 2021

@author: colem
"""
import tensorflow as tensorflow
import tensorflow.keras as tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk as nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#import tidytext as tidytext
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
nltk.download("stopwords")
stop = stopwords.words("english")
from string import punctuation
from collections import defaultdict
from pathlib import Path
from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB



#reading in dataframe ans separating sentences column
SOT = pd.read_csv("sentencesWithPolarity2.csv")
sentences = SOT["Sentences"]

#Tokenize the sentences
myTokenizer = Tokenizer(num_words=10000)
myTokenizer.fit_on_texts(sentences)


#Splitting data
data = myTokenizer.texts_to_sequences(sentences)
labels = SOT["Labels"] 

train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_train[0]
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

from tensorflow import keras
from tensorflow.keras import layers
#model definition
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="softmax")
])
#compiling model
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
#Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#Train your model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
history_dict.keys()
#Plotting training data and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
#Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
#Retraining the model
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="softmax")
])


model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}
NBmodel = MultinomialNB()
NBmodel.fit(x_train, y_train)
acc = NBmodel.score(x_test, y_test)
val_acc=NBmodel.score(x_val,y_val)
print("NBMODEL Acc: " , acc)
print("NBMODEL Validation Acc: " , val_acc)


BNBmodel = BernoulliNB()
BNBmodel.fit(x_train, y_train)
acc = BNBmodel.score(x_test, y_test)
val_acc=BNBmodel.score(x_val,y_val)
print("BNBMODEL Acc: " , acc)
print("BNBMODEL Validation Acc: " , val_acc)

CNBmodel = ComplementNB()
CNBmodel.fit(x_train, y_train)
acc = CNBmodel.score(x_test, y_test)
val_acc=CNBmodel.score(x_val,y_val)
print("CNBMODEL Acc: " , acc)
print("CNBMODEL Validation Acc: " , val_acc)

KNmodel = KNeighborsClassifier()
KNmodel.fit(x_train, y_train)
acc = KNmodel.score(x_test, y_test)
val_acc=KNmodel.score(x_val,y_val)
print("KNMODEL Acc: " , acc)
print("KNMODEL Validation Acc: " , val_acc)

MLPmodel = MLPClassifier()
MLPmodel.fit(x_train, y_train)
acc = MLPmodel.score(x_test, y_test)
val_acc=MLPmodel.score(x_val,y_val)
print("MLPMODEL Acc: " , acc)
print("MLPMODEL Validation Acc: " , val_acc)

ADAmodel = AdaBoostClassifier()
ADAmodel.fit(x_train, y_train)
acc = ADAmodel.score(x_test, y_test)
val_acc=ADAmodel.score(x_val,y_val)
print("ADAMODEL Acc: " , acc)
print("ADAMODEL Validation Acc: " , val_acc)

DTmodel = DecisionTreeClassifier()
DTmodel.fit(x_train, y_train)
acc = DTmodel.score(x_test, y_test)
val_acc=DTmodel.score(x_val,y_val)
print("DTMODEL Acc: " , acc)
print("DTMODEL Validation Acc: " , val_acc)

RFmodel = RandomForestClassifier()
RFmodel.fit(x_train, y_train)
acc = RFmodel.score(x_test, y_test)
val_acc=RFmodel.score(x_val,y_val)
print("RFMODEL Acc: " , acc)
print("RFMODEL Validation Acc: " , val_acc)

LRmodel = LogisticRegression()
LRmodel.fit(x_train, y_train)
acc = LRmodel.score(x_test, y_test)
val_acc=LRmodel.score(x_val,y_val)
print("LRMODEL Acc: " , acc)
print("LRMODEL Validation Acc: " , val_acc)