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

###Inverse Class Frequency###
def inverse_class_frequency(pos_documents,neg_documents,vocabulary_list):
	N=2
	ICF=[]
	for word in vocabulary_list:
		count_pos=0
		count_neg=0
		for document in pos_documents:
			if word in document:
				count_pos=1
		for document in neg_documents:
			if word in document:
				count_neg=1
		count=count_pos+count_neg
		icf=math.log(N/count,2)
		ICF.append(icf)
	return ICF


###Term Frequency Inverse Class Frequency###
def term_frequency_inverse_class_frequency(TF,ICF):
	ICF=np.array(ICF)
	TFICF=[]
	for tf in TF:
		tf=np.array(tf)
		tficf=tf*ICF
		TFICF.append(tficf.tolist())
	return TFICF


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


###Performing TFICF###
TF=term_frequency(documents,vocabulary_list)
ICF=inverse_class_frequency(pos_documents,neg_documents,vocabulary_list)
TFICF=term_frequency_inverse_class_frequency(TF,ICF)

###Output###
print("Vocabulary to be considered: ")
print(vocabulary_list)
print()
print("Term Frequency wrt Vocabulary: ")
print(TF)
print()
print("Inverse Class Frequency wrt Vocabulary: ")
print(ICF)
print()
print("Term Frequency Inverse Class Frequency: ")
print(TFICF)

file_path=sys.argv[1]+'/tficf.txt'
f = open(file_path, 'w')
simplejson.dump(TFICF, f)
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
TF=term_frequency(doc,i_n)
ICF=inverse_class_frequency(pos_doc,neg_doc,i_n)
TFICF=term_frequency_inverse_class_frequency(TF,ICF)

print("Vocabulary to be considered: ")
print(i_n)
print()
print("Term Frequency wrt Vocabulary: ")
print(TF)
print()
print("Inverse Class Frequency wrt Vocabulary: ")
print(ICF)
print()
print("Term Frequency Inverse Class Frequency: ")
print(TFICF)
"""