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

###CHI for Positive Corpus###
def CHI_for_positive_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq):
	pos_D=len(pos_documents)
	neg_D=len(neg_documents)
	D=pos_D+neg_D
	CHI_pos=[]
	for i in range(len(pos_documents_freq)):
		ncti=pos_documents_freq[i]
		ncbtib=neg_D - neg_documents_freq[i]
		ncbti=neg_documents_freq[i]
		nctib=pos_D - pos_documents_freq[i]


		numerator = D * math.pow((ncti * ncbtib - ncbti * nctib ),2)
		denominator=(ncti+nctib)*(ncbti+ncbtib)*(ncti+ncbti)*(nctib+ncbtib)

		if denominator == 0:
			CHI_per_term=0
		else:
			CHI_per_term = float(numerator)/denominator

		CHI_pos.append(CHI_per_term)
	return CHI_pos

###CHI for Negative Corpus###
def CHI_for_negative_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq):
	pos_D=len(pos_documents)
	neg_D=len(neg_documents)
	D=pos_D+neg_D
	CHI_neg=[]
	for i in range(len(neg_documents_freq)):
		ncti=neg_documents_freq[i]
		ncbtib=pos_D - pos_documents_freq[i]
		ncbti=pos_documents_freq[i]
		nctib=neg_D - neg_documents_freq[i]

		numerator =  D * math.pow((ncti * ncbtib - ncbti * nctib ),2)
		denominator=(ncti+nctib)*(ncbti+ncbtib)*(ncti+ncbti)*(nctib+ncbtib)

		if denominator == 0:
			CHI_per_term = 0
		else:
			CHI_per_term = float(numerator)/denominator

		CHI_neg.append(CHI_per_term)
	return CHI_neg

###Calculating CHI###
def CHI(CHI_pos,CHI_neg):
	CHI_result=[]
	for i in range(len(CHI_pos)):
		CHI_result.append(max(CHI_pos[i],CHI_neg[i]))

	return CHI_result


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


###Performing CHI###
pos_documents_freq,neg_documents_freq=document_frequency(pos_documents,neg_documents,vocabulary_list)
CHI_pos=CHI_for_positive_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq)
CHI_neg=CHI_for_negative_corpus(pos_documents,neg_documents,pos_documents_freq,neg_documents_freq)
CHI_result=CHI(CHI_pos,CHI_neg)


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
print("CHI for Positive Corpus: ")
print(CHI_pos)
print()
print()
print("CHI for Negative Corpus: ")
print(CHI_neg)
print()
print("CHI: ")
print(CHI_result)

file_path=sys.argv[1]+'/chi.txt'
f = open(file_path, 'w')
simplejson.dump(CHI_result, f)
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
CHI_pos=CHI_for_positive_corpus(pos_doc,neg_doc,pos_documents_freq,neg_documents_freq)
CHI_neg=CHI_for_negative_corpus(pos_doc,neg_doc,pos_documents_freq,neg_documents_freq)
CHI_result=CHI(CHI_pos,CHI_neg)


print("Vocabulary to be considered: ")
print(i_n)
print()
print("Document Frequency wrt Positive Corpus: ")
print(pos_documents_freq)
print()
print("Document Frequency wrt Negative Corpus: ")
print(neg_documents_freq)
print()
print("CHI for Positive Corpus: ")
print(CHI_pos)
print()
print()
print("CHI for Negative Corpus: ")
print(CHI_neg)
print()
print("CHI: ")
print(CHI_result)
"""


