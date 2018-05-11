# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:29:12 2018

@author: Yatao.Lu
"""

import nltk
from nltk.corpus import stopwords
##read data in and tokenize 
from nltk.tokenize import sent_tokenize, word_tokenize
 
data = open('data.txt').read()
###3
###
tokens = [t for t in data.split()]
print (tokens)
clean_tokens = tokens[:]
 
sr = stopwords.words('english')
 
for token in tokens:
 
    if token in stopwords.words('english'):
 
        clean_tokens.remove(token) 

##
###
words = word_tokenize(data)
phrases = sent_tokenize(data)

print(words)

##filter out stop words
from nltk.corpus import stopwords
sr = stopwords.words('english')
wordsfiltered = words[:]
for w in words:
 
    if w in stopwords.words('english'):
 
        wordsfiltered.remove(w) 
print(wordsfiltered)

##viz the frequency

freq = nltk.FreqDist(clean_tokens)
 
for key,val in freq.items():
 
    print (str(key) + ':' + str(val))
freq.plot(20,cumulative=False)
####deal with same word in different type
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
 
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer()  
wordTrans = []
 
for word in clean_tokens:
    wordTrans.append(lemmatizer.lemmatize(word))
    
###synonyms

#from nltk.corpus import wordnet    

#synonyms = []
#for word in wordTrans:
#    synonyms.append(wordnet.synsets(word))
    
    
#for syn in wordnet.synsets('Computer'):
 
#    for lemma in syn.lemmas():
 
#        synonyms.append(lemma.name())
 
#print(synonyms)


###tagging the word into 
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
document = open('data.txt').read()
sentences = nltk.sent_tokenize(document)   
 
data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))
 
for word in data: 
    if 'JJ' in word[0]: 
        print(word)

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
document = open('data.txt').read()
sentences = nltk.sent_tokenize(document)   
data_raw = [] 
for sent in sentences:
    data_raw = data_raw + nltk.word_tokenize(sent)
    
    
import nltk.corpus
from nltk.text import Text
textList = Text(data_raw)
textList.concordance('best')

