import re
import pandas as pd
import numpy as np
import math
import random
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

from nltk.metrics import ConfusionMatrix
from nltk.corpus import stopwords
from nltk.corpus import genesis
from nltk.corpus import udhr

def lexical_diversity(text):
    return len(set(text)) / len(text)

def createCorpus():
    english = genesis.words("english-kjv.txt")
    english_web = genesis.words("english-web.txt")
    finnish = genesis.words("finnish.txt")
    french = genesis.words("french.txt")
    portuguese = genesis.words("portuguese.txt")

    languages = ['English-Latin1', 'German_Deutsch-Latin1', 'French_Francais-Latin1', 'Spanish-Latin1']

    d = {'genesis_corpus': [english, english_web, french, finnish,portuguese], 'language': [1,1, 0,0,0]}
    for language in languages:
        if language != 'English-Latin1':
            d['genesis_corpus'].append(udhr.words(language))
            d['language'].append(0)
        else:
            d['genesis_corpus'].append(udhr.words(language))
            d['language'].append(1)
    df =pd.DataFrame(data=d)

    print('\nInformation about the corpus.\n')

    print(f'Information about genesis corpus in english.\n Length: {len(english)}, Lexical diversity: {lexical_diversity(english)}')
    print(f'Information about genesis corpus in english (web).\n Length: {len(english_web)}, Lexical diversity: {lexical_diversity(english_web)}')
    print(f'Information about genesis corpus in finnish.\n Length: {len(finnish)}, Lexical diversity: {lexical_diversity(finnish)}')
    print(f'Information about genesis corpus in french.\n Length: {len(french)}, Lexical diversity: {lexical_diversity(french)}')
    print(f'Information about genesis corpus in portuguese.\n Length: {len(portuguese)}, Lexical diversity: {lexical_diversity(portuguese)}')
    print(f"Information about udhr corpus in english-latin1.\n Length: {len(df['genesis_corpus'][5])}, Lexical diversity: {lexical_diversity(df['genesis_corpus'][5])}")
    print(f"Information about udhr corpus in german.\n Length: {len(df['genesis_corpus'][6])}, Lexical diversity: {lexical_diversity(df['genesis_corpus'][6])}")
    print(f"Information about udhr corpus in french.\n Length: {len(df['genesis_corpus'][7])}, Lexical diversity: {lexical_diversity(df['genesis_corpus'][7])}")
    print(f"Information about udhr corpus in spanish.\n Length: {len(df['genesis_corpus'][8])}, Lexical diversity: {lexical_diversity(df['genesis_corpus'][8])}") 
    
    return df

def remove_nonalpha(string):
    results = [word for word in string if re.match(r'[a-zA-Z]+',word)]
    return results

def remove_stopwords(string):
    stop_words = set(stopwords.words('english') + stopwords.words('french') + 
    stopwords.words('finnish') + stopwords.words('portuguese') + 
    stopwords.words('german')+ stopwords.words('spanish'))
    results = []
    for word in string:
        if word not in stop_words: results.append(word.lower())
    return ' '.join(results)

def bagofwords(corpus):
    wordfreq = {}
    tokens = w_tokenizer.tokenize(corpus)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
    return wordfreq

########### STARTING OF MAIN ###########
print('Starting of main...\n')
df = createCorpus()
df['genesis_corpus']=df['genesis_corpus'].apply(lambda cw : remove_nonalpha(cw))
df['genesis_corpus']=df['genesis_corpus'].apply(lambda cw : remove_stopwords(cw))
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
df['genesis_corpus'] = df['genesis_corpus'].apply(lambda cw : bagofwords(cw))

no_eng = list(np.where(df['language'] == 0)[0])
eng = list(np.where(df['language'] == 1)[0])

labeled_corpusno = [
    ({word:freq}, 'non-english') 
    for corp in df.iloc[no_eng]['genesis_corpus']
    for word,freq in corp.items()
]
labeled_corpuseng = [
    ({word:freq}, 'english') 
    for corp in df.iloc[eng]['genesis_corpus']
    for word,freq in corp.items()
    ]
featureset = labeled_corpuseng + labeled_corpusno
random.shuffle(featureset)

train = featureset[:math.ceil(2*(len(featureset)/3))]
test = featureset[math.ceil(2*(len(featureset)/3)):]
print(f'\nInformation about the featureset.\nLength of the training set: {len(train)}\nLength of the testing set: {len(test)}\n')

#Classifying
print('\nTraining the classifier...\n')
classifier = nltk.NaiveBayesClassifier.train(train)
print('End of training. Showing the 15 most informative features:\n')
classifier.show_most_informative_features(15)
print('\nStart of testing...\n')
test_wolabes = [item[0] for item in test]
test_classified = classifier.classify_many(test_wolabes)
reference = [elem[1]
    for elem in test 
]
print(f'Accuracty reached: {nltk.classify.accuracy(classifier, test)}')
print('\nCreating the confusion matrix...\n')
cm = ConfusionMatrix(reference,test_classified)
print(cm)
print(f'\nPrecision of english classification: {cm.precision("english")}')
print(f'\nPrecision of non-english classification: {cm.precision("non-english")}')
print(f'\nRecall of english classification: {cm.recall("english")}')
print(f'\nRecall of non-english classification: {cm.recall("non-english")}')
print(f'\nF-score of english classification: {cm.f_measure("english")}')
print(f'\nF-score of non-english classification: {cm.f_measure("non-english")}')
print('\nEnd of main.')