# Comments on the first assignment

## Introduction

The classifier was developed with the nltk package, using its own classifier nltk.NaiveBayesClassifier() with a simple pipeline, that resembles word2vec:

- Acquisition of data
- Cleaning and pre-processing:
  - Removal of non-alphanumeric characters and words
  - Removal of stop-words of all the languages
- Tokenization
- Creation of the Bag of Words
- Splitting training and testing sets
- Training the model
- Testing and Querying the model

The corpus was made mixing 9 pre-existing nltk corpus, from the Genesis corpus and the Universal declaration of human rights corpus:

- 3 of the corpus are in english (two from the genesis and one from the Udhr)
- The other languages used are: Finnish, French (2 corpus), Portuguese, German and Spanish

## Size of the corpus

Information about the corpus.
(Where 1 is english and 0 is non-english)

- Information about corpus number 0\
 Length: 44764, Lexical diversity: 0.06230453042623537, Language: 1
- Information about corpus number 1\
 Length: 44054, Lexical diversity: 0.06033504335588142, Language: 1
- Information about corpus number 2\
 Length: 32520, Lexical diversity: 0.2088560885608856, Language: 0
- Information about corpus number 3\
 Length: 46116, Lexical diversity: 0.0803842484170353, Language: 0
- Information about corpus number 4\
 Length: 45094, Lexical diversity: 0.08457887967357076, Language: 0
- Information about corpus number 5\
 Length: 192427, Lexical diversity: 0.04059201671283136, Language: 1
- Information about corpus number 6\
 Length: 23140, Lexical diversity: 0.17359550561797754, Language: 1
- Information about corpus number 7\
 Length: 1781, Lexical diversity: 0.29927007299270075, Language: 1
- Information about corpus number 8\
 Length: 1521, Lexical diversity: 0.3806706114398422, Language: 0
- Information about corpus number 9\
 Length: 1935, Lexical diversity: 0.2930232558139535, Language: 0
- Information about corpus number 10\
 Length: 1763, Lexical diversity: 0.3074305161656268, Language: 0

## Creating training and testing set

Common split percentages include:

- Train: 80%, Test: 20%
- Train: 67%, Test: 33%
- Train: 50%, Test: 50%

Given the size of the corpus and some experiments during the developing of the code, it's been chosen to use the 67% and 33% split of the dataset.\
In the end, after the preprocessing and cleaning of the data, the size of the train set is 19574 words and the size of the test set is 10514 words.

## Performance indicators

The performance of the classifier on the constructed test set is analyzed with confusion matrix by measuring the standard
metrics that are commonly used for measuring the classification performance of other classification models.\
The experiments are evaluated using the standard metrics of accuracy, precision, recall and F-measure for classification.
These were calculated using the predictive classification table, known as Confusion Matrix, where:

- TN (True Negative) : Number of correct predictions that an instance is irrelevant
- FP (False Positive) : Number of incorrect predictions that an instance is relevant
- FN (False Negative) : Number of incorrect predictions that an instance is irrelevant
- TP (True Positive) : Number of correct predictions that an instance is relevant
- Accuracy(ACC) - The proportion of the total number of predictions that were correct:
  - Accuracy (\%) = (TN + TP)/(TN+FN+FP+TP)
- Precision(PREC) - The proportion of the predicted relevant materials data sets that were correct:
  - Precision (\%) = TP / (FP + TP)
- Recall(REC) - The proportion of the relevant materials data sets that were correctly identified
  - Recall (\%) = TP / (FN + TP)
- F-Measure(FM) - Derives from precision and recall values:
  - F-Measure (\%) = (2 x REC x PREC)/(REC + PREC)

## Employability

Each row as to be intended as the one in which the label is the True statement (so we'll look at the first one). Given the F-measure though I don't think it is actually employable as classifier.
