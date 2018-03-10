import sys
import ast
from collections import Counter
from os import listdir
import simplejson
import math
import numpy as np

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

###Term Frequency According to paper###
def term_frequency_paper(documents,vocabulary_list):
	TF=[]

	for document in documents:
		N=len(document)

		tf_per_document=[]
		for word in vocabulary_list:
			fij=document.count(word)

			tf=float(fij)/N
			tf_per_document.append(tf)
		TF.append(tf_per_document)

	return TF

###Inverse Document Frequency###
def inverse_document_frequency(documents,vocabulary_list):
	IDF=[]
	N=len(documents)
	for word in vocabulary_list:
		count=0
		for document in documents:
			if word in document:
				count+=1
		idf=math.log(N/count,2)
		IDF.append(idf)
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


"""
###Accessing the Vocabulary###
file_path=sys.argv[1]+'/vocabulary.txt'
f1 = open(file_path, 'r')
vocabulary_list=f1.read()
vocabulary_list = ast.literal_eval(vocabulary_list)
f1.close()

###Accessing the Documents###
pos_directory_path="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/"+sys.argv[1]+"/Cleaning_Stopword_Removal/pos"
#pos_directory_path=['Absolute_Path_To_This_Directory']+sys.argv[1]+"/Cleaning_Stopword_Removal/pos"
neg_directory_path="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/"+sys.argv[1]+"/Cleaning_Stopword_Removal/neg"
#neg_directory_path=['Absolute_Path_To_This_Directory']+sys.argv[1]+"/Cleaning_Stopword_Removal/neg"
documents=[]

for file in listdir(pos_directory_path):
	file_path=pos_directory_path+"/"+file
	f=open(file_path,'r')
	document=f.read()
	document = ast.literal_eval(document)
	documents.append(document)

for file in listdir(neg_directory_path):
	file_path=neg_directory_path+"/"+file
	f=open(file_path,'r')
	document=f.read()
	document = ast.literal_eval(document)
	documents.append(document)



###Performing TFIDF###
TF=term_frequency(documents,vocabulary_list)
IDF=inverse_document_frequency(documents,vocabulary_list)
TFIDF=term_frequency_inverse_document_frequency(TF,IDF)

file_path=sys.argv[1]+'/tfidf.txt'
f = open(file_path, 'w')
simplejson.dump(TFIDF, f)
f.close()

###Output###
print("Vocabulary to be considered: ")
print(vocabulary_list)
print()
print("Term Frequency wrt Vocabulary: ")
print(TF)
print()
print("Inverse Document Frequency wrt Vocabulary: ")
print(IDF)
print()
print("Term Frequency Inverse Document Frequency: ")
print(TFIDF)
"""


"""
Trivial Example:
----------------
"""

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

print("Vocabulary to be considered: ")
print(i_n)
print()
print("Term Frequency wrt Vocabulary: ")
print(TF)
print()
print("Inverse Document Frequency wrt Vocabulary: ")
print(IDF)
print()
print("Term Frequency Inverse Document Frequency: ")
print(TFIDF)

