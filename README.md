# Semantic-Analysis-AI-Final-Project

Notes -
* _The following project was completed for the course CSC 434 Artificial Intelligence and Machine Learning, offered at The College of Brockport._
* _Citations are referred to in brackets [#]._

Date Submitted: 12/2021, Fall Semester

Build Status: ![example param](https://github.com/github/docs/actions/workflows/main.yml/badge.svg?branch=main)

**Project Target**: The goal of this project is to build a Natural Language Processing (NLP) model which utilizes multiclass classification to perform the task of semantic analysis upon a set of data.

Built with -
* Tensorflow

Associated Packages - 
* Keras
* nltk
* pandas

Dataset Source -
* Kaggle

Dataset Used - 
* State of The Union Corpus (1790-2018) [5]

The code written was heavily referenced from the textbook, _"Deep learning with python"_ by Francois Chollet [1].

Overview: Following consecutive steps, building the desired NLP model involves importing a dataset, preprocessing through data tokenization, encoding, vectorization, creation of a model, and finally validation of the built model, thus meeting target requirements. Transformations done upon the original dataset are done in _sentence_df.py_ while preprocessing the data through vectorization into tensors as well as building, plotting, and validation of the model is done in _SOU.py_.

REFERENCES:
1. Chollet, Francois, “Getting started with neural networks: Classification and regression,” in Deep learning with python, S.l., CA: O'REILLY MEDIA, 2021.
2. M. Mogyorosi, “Sentiment analysis: First steps with Python's NLTK library,” Real Python, 24-Sep-2021. [Online]. Available: https://realpython.com/python-nltk-     sentiment-analysis/. [Accessed: 19-Nov-2021]. 
3. R. F. Baumeister and K. D. Vohs, “Content Analysis,” Encyclopedia of Social Psychology. SAGE, London, 2007. 
4. NobelNobel, cs95 , jpp, Heraknos, BENY , and c z, “Split cell into multiple rows in   pandas dataframe,” Stack Overflow, 01-Jul-1966. [Online]. Available: https://stackoverflow.com/questions/50731229/split-cell-into-multiple-rows-in-pandas-dataframe. [Accessed: 17-Nov-2021]. 
5. R. Tatman, “State of the Union Corpus (1790 - 2018),” Kaggle, 19-Oct-2018. [Online]. Available: https://www.kaggle.com/rtatman/state-of-the-union-corpus-1989-2017. [Accessed: 11-Oct-2021]. 
6. Sulphix, D. Dotterel, and S. Scarab, “Remove punctuation in dataframe column code example,” remove punctuation in dataframe column Code Example. [Online]. Available: https://www.codegrepper.com/code-examples/python/remove+punctuation+in+dataframe+column. [Accessed: 17-Nov-2021]. 

CREDITS:
coleman3616, dhodzic1, kyle-knopp, mwarren585
