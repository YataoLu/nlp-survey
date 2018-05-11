# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:01:29 2018

@author: Yatao.Lu
"""
###############################
#read and tokenize the data
import nltk

document = open('data.txt').read()
sentences = nltk.sent_tokenize(document)   
data_raw = [] 
for sent in sentences:
    data_raw = data_raw + nltk.word_tokenize(sent)
    
######################################
#lower case
# convert to lower case
lowercase = [w.lower() for w in data_raw]

################################
#remove puntuation
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in lowercase]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])

# stemming of words
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
stemmed = [lemmatizer.lemmatize(word) for word in words]

###############
#viz the frequency
freq = nltk.FreqDist(stemmed)
 
for key,val in freq.items():
 
    print (str(key) + ':' + str(val))
freq.plot(20,cumulative=False)

##########################
#see the sencentece with loan
import nltk.corpus
from nltk.text import Text
textList = Text(stemmed)
print(textList)
textList.concordance('loan')

#############################
#filter only adj. , or adj. along with loan
import pandas as pd
data=[]
data += nltk.pos_tag(stemmed)
adj=[]
for w in data: 
        if w[1] == 'JJ' or w[1] == 'JJR' or w[1] == 'JJS' :
            adj += w
            only_words = list(filter(lambda x: (x != 'JJ') and (x != 'JJR') and (x != 'JJS'), adj))
            print(only_words)

df2= pd.DataFrame(only_words)
df2.columns = ['word']
df2['word'].value_counts()[:10].plot(kind='bar')




from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000,
                                 min_df=0.0001, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(stemmed) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()


 



from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)


from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()








import string
import collections
 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

import random
file=stemmed
random.shuffle(file)
train_data = file[:int((len(data)+1)*.80)] #Remaining 80% to training set
test_data = file[int(len(data)*.80+1):] #Splits 20% data to test set

###################################################
#Extracting features from text files
#bags of words
#Term Frequency times inverse document frequency.
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """

    km_model = KMeans(n_clusters=clusters)
    km_model.fit(X_train_tfidf)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering
if __name__ == "__main__":
    articles = [...]
    clusters = cluster_texts(articles, 7)
    pprint(dict(clusters))



#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(7):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % stemmed.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
    
print()
print()




# output: [0, 2, 1, 2, 2, 1, 2, 2, 0, 1, 0, 1, 2, 1, 2]
 
train_data[91,
     142,
     200,
     217,
     287,
     325,
     506,
     677,
     697,
     850,
     980,
     1109,
     1131,
     1168,
     1219,
     1325,
     1392,
     1419,
     1444,
     1751,
     1757,
     1758,
     1802,
     1890]


#################
#separate data to training and test data
import random
file=stemmed
random.shuffle(file)
train_data = file[:int((len(data)+1)*.80)] #Remaining 80% to training set
test_data = file[int(len(data)*.80+1):] #Splits 20% data to test set

###################################################
#Extracting features from text files
#bags of words
#Term Frequency times inverse document frequency.
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train_data)

########################################################
#fit naive bayes
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(train_data, train_data)
#Performance of NB Classifier
import numpy as np
predicted = text_clf.predict(test_data)
np.mean(predicted == test_data)

####################################################
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3, n_iter=5, random_state=42)),
                                                   ])
text_clf_svm = text_clf_svm.fit(train_data, train_data)
predicted_svm = text_clf_svm.predict(test_data)
np.mean(predicted_svm == test_data)

##########################################################
#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train_data, train_data)
gs_clf.best_score_
gs_clf.best_params_

# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}


gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(train_data, train_data)
gs_clf_svm.best_score_
gs_clf_svm.best_params_






# NLTK
# Removing stop words
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])


# Stemming Code
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(train_data, train_data[62])

predicted_mnb_stemmed = text_mnb_stemmed.predict(test_data)

np.mean(predicted_mnb_stemmed == test_data)






