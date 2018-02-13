from os import listdir
from nltk import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from collections import Counter
import math
import numpy as np

def loading_all_files(directory_path):
	for file in listdir(directory_path):
		file_path=directory_path+"/"+file
		load_a_file(file_path)

def load_a_file(file_path):
	file=open(file_path,'r')
	data=file.read()
	tokenization(data)
	file.close()

###Step1 of Preprocessing---Tokenization###
def tokenization(data):
	token_list=data.split()
	documents.append(token_list)

###Step2 of Preprocessing---Normalization(It can be assumed normalized)###

###Step3 of Preprocessing---Stemming###
def stemming(documents):
	for document in documents:
		for i in range(len(document)):
			stemming_token=PorterStemmer().stem(document[i])
			document[i]=stemming_token

###Step4 of Preprocessing---Cleaning & Stopword Removal###
'''
Cleaning should have the following steps:
(i) Removing tokens that contains punctuations
(ii) Removing tokens that are just punctuations
(iii) Removing tokens that contain numbers
(iv) Removing tokens that have one character 
'''
def cleaning_and_stopword_removal(documents):
	###(i) & (ii)###
	translation = str.maketrans("","", string.punctuation);
	for document in documents:
		for i in range(len(document)):
			document[i]=document[i].translate(translation);

	###(iii)###
	documents=[[token for token in document if token.isalpha()]for document in documents]

	###(iv)###
	documents=[[token for token in document if len(token)>1]for document in documents]

	###Stopword Removal###
	stop_words = set(stopwords.words('english'))
	documents=[[token for token in document if not token in stop_words]for document in documents]
	return documents

###Creating a vocabulary###
def create_vocabulary(documents):
	for document in documents:
		vocabulary.update(document)
	vocabulary_list = [word for word,frequency in vocabulary.items() if frequency >= 5]
	return vocabulary_list

###Term Frequency###
def term_frequency(documents,vocabulary_list):
	TF=[]
	for document in documents:
		tf_per_document=[]
		for word in vocabulary_list:
			fij=document.count(word)
			if fij>0:
				tf=1+math.log(fij,2)
			else:
				tf=0
			tf_per_document.append(tf)
		TF.append(tf_per_document)

	return TF

###Inverse Document Frequency###
def inverse_document_frequency(documents,vocabulary_list):
	count=0
	IDF=[]
	N=len(documents)
	for word in vocabulary_list:
		for document in documents:
			if word in document:
				count+=1
		idf=math.log(N/count,2)
		IDF.append(idf)
		count=0
	return IDF

###Term Frequency Inverse Document Frequency###
def term_frequency_inverse_document_frequency(TF,IDF):
	IDF=np.array(IDF)
	TFIDF=[]
	for tf in TF:
		tf=np.array(tf)
		tfidf=tf*IDF
		TFIDF.append(tfidf.tolist())
	return TFIDF



###Initializing###
documents=[]
vocabulary=Counter()
directory_path_for_positive_files="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/pos"
directory_path_for_negative_files="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/neg"
loading_all_files(directory_path_for_positive_files)
loading_all_files(directory_path_for_negative_files)
stemming(documents)
documents=cleaning_and_stopword_removal(documents)
vocabulary_list=create_vocabulary(documents)

###TFIDF###
TF=term_frequency(documents,vocabulary_list)
IDF=inverse_document_frequency(documents,vocabulary_list)
TFIDF=term_frequency_inverse_document_frequency(TF,IDF)
print(TFIDF)


"""
TESTING
doc=[['to', 'do', 'is', 'to', 'be', 'to', 'be', 'is', 'to', 'do'],
 ['to', 'be', 'or', 'not', 'to', 'be', 'i', 'am', 'what', 'i', 'am'],
 ['i', 'think', 'therefore', 'i', 'am', 'do', 'be', 'do', 'be', 'do'],
 ['do', 'do', 'do', 'da', 'da', 'da', 'let', 'it', 'be', 'let', 'it', 'be']]

i_n=['to',
 'do',
 'is',
 'be',
 'or',
 'not',
 'i',
 'am',
 'what',
 'think',
 'therefore',
 'da',
 'let',
 'it']
TF=term_frequency(doc,i_n)
IDF=inverse_document_frequency(doc,i_n)
TFIDF=term_frequency_inverse_document_frequency(TF,IDF)
print(TFIDF)
"""
