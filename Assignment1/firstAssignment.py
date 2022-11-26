import re
import pandas as pd
import numpy as np
import math
import random

from collections import Counter
from nltk import NaiveBayesClassifier 
from nltk.classify import accuracy
from nltk.tokenize import WhitespaceTokenizer
from nltk.metrics import ConfusionMatrix
from nltk.corpus import stopwords
from nltk.corpus import genesis
from nltk.corpus import udhr
from nltk.corpus import gutenberg

def lexical_diversity(text):
    return len(set(text)) / len(text)

def createCorpus():
    d = { 'corpus':
            [genesis.words("english-kjv.txt"),genesis.words("english-web.txt"),
                genesis.words("finnish.txt"),genesis.words("french.txt"),genesis.words("portuguese.txt"),
                gutenberg.words("austen-emma.txt"),gutenberg.words("shakespeare-macbeth.txt"),
                udhr.words("English-Latin1"),udhr.words("German_Deutsch-Latin1"),
                udhr.words("French_Francais-Latin1"),udhr.words("Spanish-Latin1")
            ],
            'language':
            [1,1,0,0,0,1,1,1,0,0,0]
    }
    df = pd.DataFrame(data=d)
    print('\nInformation about the corpus.\n(Where 1 is english and 0 is non-english)\n')

    for index,corpus in df.iterrows():
        print(f"Information about corpus number {index}\n Length: {len(corpus['corpus'])}, Lexical diversity: {lexical_diversity(corpus['corpus'])}, Language: {corpus['language']}")
    
    return df

def remove_nonalpha(string):
    return [word for word in string if re.match(r'[a-zA-Z]+',word)]

def remove_stopwords(string):
    stop_words = set(stopwords.words('english') + stopwords.words('french') + 
        stopwords.words('finnish') + stopwords.words('portuguese') + 
        stopwords.words('german')+ stopwords.words('spanish'))
    return ' '.join(word.lower() for word in string if word not in stop_words)

def bagofwords(corpus):
    w_tokenizer = WhitespaceTokenizer()
    return Counter(token for token in w_tokenizer.tokenize(corpus))

########### STARTING OF MAIN ###########
print('Starting of main...\n')
df = createCorpus()
df['corpus']=df['corpus'].apply(lambda cw : remove_nonalpha(cw))
df['corpus']=df['corpus'].apply(lambda cw : remove_stopwords(cw))
df['corpus'] = df['corpus'].apply(lambda cw : bagofwords(cw))

labeled_corpusno = [
    ({word:freq}, 'non-english') 
    for corp in df[df['language'] == 0]['corpus']
    for word,freq in corp.items()
]
labeled_corpuseng = [
    ({word:freq}, 'english') 
    for corp in df[df['language'] == 1]['corpus']
    for word,freq in corp.items()
]
featureset = labeled_corpuseng + labeled_corpusno
random.shuffle(featureset)

train = featureset[:math.ceil(2*(len(featureset)/3))]
test = featureset[math.ceil(2*(len(featureset)/3)):]
print(f'\nInformation about the featureset.\nLength of the training set: {len(train)}\nLength of the testing set: {len(test)}\n')

#Classifying
print('\nTraining the classifier...\n')
classifier = NaiveBayesClassifier.train(train)
print('End of training. Showing the 15 most informative features:\n')
classifier.show_most_informative_features(15)
print('\nStart of testing...\n')
test_wolabes = [item[0] for item in test]
test_classified = classifier.classify_many(test_wolabes)
reference = [elem[1]
    for elem in test 
]
print(f'Accuracty reached: {accuracy(classifier, test)}')
print('\nCreating the confusion matrix...\n')
cm = ConfusionMatrix(reference,test_classified)
print(cm)
print(cm.evaluate())
print('\nEnd of main.')