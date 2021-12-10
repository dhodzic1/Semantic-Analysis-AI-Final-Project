import nltk as nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from pandas import DataFrame

stop = stopwords.words('english')
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import string
import numpy as np
from string import punctuation
from collections import defaultdict
from pathlib import Path
from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

dir_path = "archive/sotu"


# Reading the SOTU data into a dataframe
results = defaultdict(list)
for file in Path(dir_path).iterdir():
    with open(file, "r") as file_open:
        results["file_name"].append(file.name)
        results["text"].append(file_open.read())
df1: DataFrame = pd.DataFrame(results)


# new data frame with split value columns which splits column at underscore
newDF = df1["file_name"].str.split("_", n=1, expand=True)

# new data frame which splits column at period
newDF1 = newDF[1].str.split(".", n=1, expand=True)

# Removinng punctuations and stop words, replacing next line symbols and double spaces with single space
df1["text"] = df1["text"].str.replace("\n", " ")  # replace newlines


# creating new data frame organized by President, year and speech
data = [newDF[0], newDF1[0], df1["text"]]
headers = ["President", "Year", "Speech"]
stateOfUnion = pd.concat(data, axis=1, keys=headers)

# spliting speach into individual words
sentences = (stateOfUnion.set_index(['President', 'Year']).apply(lambda x: x.str.split('.').explode()).reset_index())
#print(sentences)

#sentences.to_csv('sentences.csv')

# pass text into function defined here to remove all punctuation
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text

sentences["Speech"] = sentences["Speech"].apply(remove_punctuations)  # remove punctuation
sentences["Speech"] = sentences["Speech"].str.replace("  ", " ")
sentences["Speech"] = sentences["Speech"].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))

#print(sentences)

#sentences.to_csv('sentences2.csv')

#sentences["Speech"].to_csv('onlySentences.csv')

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
ph = 0
listPolarity = []
for text in sentences["Speech"]:
    polarity = sia.polarity_scores(sentences["Speech"][ph])
    listPolarity.append(polarity)
    ph = ph + 1

print(listPolarity)

dfPolarity = pd.DataFrame(listPolarity)

# Creating a single list and then dataframe from the vader scores
# -1 is negative, 0 is neutral, 1 is positive
positivity_score = []
for r in listPolarity:
    if r['neg'] > 0 and r['neg']>r['neu'] and r['neg']>r['pos']:
        positivity_score.append("-1")
    elif r['neu'] > 0 and r['neu'] > r['neg'] and r['neu'] > r['pos']:
        positivity_score.append("0")
    elif r['pos'] > 0 and r['pos'] > r['neu'] and r['pos'] > r['neg']:
        positivity_score.append("1")
    else:
        positivity_score.append(pd.NA)


dfPositivity = pd.DataFrame(positivity_score)

# creating data frame with the polarity for each sentence in the SOTU speeches
data2 = [sentences.Speech, dfPositivity]
headers2 = ["Sentences", "Polarity"]
sentencesWithPolarity = pd.concat(data2, axis=1, keys=headers2)

sentencesWithPolarity.dropna(axis=0,how='any',inplace=True)

#print(sentencesWithPolarity)
sentencesWithPolarity.to_csv('sentencesWithPolarity.csv')