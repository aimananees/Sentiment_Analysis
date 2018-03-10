import sys
import ast
from collections import Counter
from os import listdir
import simplejson
import math
import numpy as np

###To find out frequency of documents that contain a particular term in the vocabulary###
def document_frequency(pos_documents,neg_documents,vocabulary_list):
	pos_documents_freq=[]
	neg_documents_freq=[]
	for word in vocabulary_list:
		pos_count=0
		neg_count=0
		for document in pos_documents:
			if word in document:
				pos_count+=1
		pos_documents_freq.append(pos_count)

		for document in neg_documents:
			if word in document:
				neg_count+=1
		neg_documents_freq.append(neg_count)

	return pos_documents_freq,neg_documents_freq

###MI for Positive Corpus###
def MI_for_positive_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq):
	pos_D=len(pos_documents)
	neg_D=len(neg_documents)
	D=pos_D+neg_D
	MI_pos=[]
	for i in range(len(pos_documents_freq)):
		numerator=pos_documents_freq[i] * D
		denominator=(pos_documents_freq[i]+neg_documents_freq[i])*len(pos_documents)

		if denominator == 0 or float(numerator)/denominator == 0:
			MI_pos.append(0)
		else:
			MI_per_term = float(numerator)/denominator
			MI_per_term=math.log(MI_per_term,2)
			MI_pos.append(MI_per_term)

	return MI_pos

###MI for Negative Corpus###
def MI_for_negative_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq):
	pos_D=len(pos_documents)
	neg_D=len(neg_documents)
	D=pos_D+neg_D
	MI_neg=[]
	for i in range(len(neg_documents_freq)):
		numerator=neg_documents_freq[i] * D
		denominator=(pos_documents_freq[i]+neg_documents_freq[i])*len(neg_documents)

		if denominator == 0 or float(numerator)/denominator == 0:
			MI_neg.append(0)
		else:
			MI_per_term = float(numerator)/denominator
			MI_per_term=math.log(MI_per_term,2)
			MI_neg.append(MI_per_term)

	return MI_neg

###Calculating MI###
def MI(MI_pos,MI_neg):
	MI_result=[]
	for i in range(len(MI_pos)):
		MI_result.append(max(MI_pos[i],MI_neg[i]))

	return MI_result

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

pos_documents=[]
neg_documents=[]
documents=[]

for file in listdir(pos_directory_path):
	file_path=pos_directory_path+"/"+file
	f=open(file_path,'r')
	document=f.read()
	document = ast.literal_eval(document)
	pos_documents.append(document)
	documents.append(document)

for file in listdir(neg_directory_path):
	file_path=neg_directory_path+"/"+file
	f=open(file_path,'r')
	document=f.read()
	document = ast.literal_eval(document)
	neg_documents.append(document)
	documents.append(document)

###Performing MI###
pos_documents_freq,neg_documents_freq=document_frequency(pos_documents,neg_documents,vocabulary_list)
MI_pos=MI_for_positive_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq)
MI_neg=MI_for_negative_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq)
MI_result=MI(MI_pos,MI_neg)

###Output###
print("Vocabulary to be considered: ")
print(vocabulary_list)
print()
print("Document Frequency wrt Positive Corpus: ")
print(pos_documents_freq)
print()
print("Document Frequency wrt Negative Corpus: ")
print(neg_documents_freq)
print()
print("MI for Positive Corpus: ")
print(MI_pos)
print()
print()
print("MI for Negative Corpus: ")
print(MI_neg)
print()
print("MI: ")
print(MI_result)

file_path=sys.argv[1]+'/mi.txt'
f = open(file_path, 'w')
simplejson.dump(MI_result, f)
f.close()

"""
Tirvial Example:
----------------


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
pos_doc=[['to', 'do', 'is', 'to', 'be', 'to', 'be', 'is', 'to', 'do'],
 ['to', 'be', 'or', 'not', 'to', 'be', 'i', 'am', 'what', 'i', 'am']]
neg_doc=[['i', 'think', 'therefore', 'i', 'am', 'do', 'be', 'do', 'be', 'do'],
 ['do', 'do', 'do', 'da', 'da', 'da', 'let', 'it', 'be', 'let', 'it', 'be']]

pos_documents_freq,neg_documents_freq=document_frequency(pos_doc,neg_doc,i_n)
MI_pos=MI_for_positive_corpus(pos_doc,neg_doc,pos_documents_freq,neg_documents_freq)
MI_neg=MI_for_negative_corpus(pos_doc,neg_doc,pos_documents_freq,neg_documents_freq)
MI_result=CHI(MI_pos,MI_neg)


print("Vocabulary to be considered: ")
print(i_n)
print()
print("Document Frequency wrt Positive Corpus: ")
print(pos_documents_freq)
print()
print("Document Frequency wrt Negative Corpus: ")
print(neg_documents_freq)
print()
print("MI for Positive Corpus: ")
print(MI_pos)
print()
print()
print("MI for Negative Corpus: ")
print(MI_neg)
print()
print("MI: ")
print(MI_result)
"""


