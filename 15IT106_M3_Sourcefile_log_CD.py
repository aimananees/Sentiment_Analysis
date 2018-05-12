import sys
import ast
from collections import Counter
from os import listdir
import simplejson
import math
import numpy as np
import csv
import pandas as pd


###Finding POS_CD of all the terms###
def POS_CD(vocabulary_list):
	file_path=sys.argv[1]+'/pos_freq.csv'
	df = pd.read_csv(file_path)
	Number_Of_Positive_Documents=len(df)
	CD_P=[]
	for word in vocabulary_list:
		saved_column = df[word]
		saved_column=saved_column.tolist()
		frequency=np.count_nonzero(saved_column)
		CD_P_Per_Term=float(frequency/Number_Of_Positive_Documents)
		CD_P.append(CD_P_Per_Term)

	return CD_P


###Finding NEG_CD of all the terms###
def NEG_CD(vocabulary_list):
	file_path=sys.argv[1]+'/neg_freq.csv'
	df = pd.read_csv(file_path)
	Number_Of_Negative_Documents=len(df)
	CD_N=[]
	for word in vocabulary_list:
		saved_column = df[word]
		saved_column=saved_column.tolist()
		frequency=np.count_nonzero(saved_column)
		CD_N_Per_Term=float(frequency/Number_Of_Negative_Documents)
		CD_N.append(CD_N_Per_Term)

	return CD_N


###Finding log_CD of all the terms###
def LOG_CD(CD_P,CD_N):
	log_CD=[]
	for i in range(len(CD_P)):
		if CD_N[i] == 0 or float(CD_P[i]/CD_N[i]) == 0:
			log_CD_Per_Term = 0
		else:
			log_CD_Per_Term = math.log(float(CD_P[i]/CD_N[i]),2)
		log_CD.append(log_CD_Per_Term)

	return log_CD


###Accessing the Vocabulary###
file_path=sys.argv[1]+'/vocabulary.txt'
f1 = open(file_path, 'r')
vocabulary_list=f1.read()
vocabulary_list = ast.literal_eval(vocabulary_list)
f1.close()


###Performing log_CD###
CD_P=POS_CD(vocabulary_list)
CD_N=NEG_CD(vocabulary_list)
log_CD=LOG_CD(CD_P,CD_N)


###Writing to CSV file###
file_path=sys.argv[1]+'/log_CD.csv'
with open(file_path, 'w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(['word','log_CD(ti)'])
	
	for i in range(len(vocabulary_list)):
		wr.writerow([vocabulary_list[i],log_CD[i]])


print("log_CD for all the words in the vocabulary_list are as follows:-")
for i in range(len(vocabulary_list)):
	print(vocabulary_list[i],'------->',log_CD[i])


"""
Tirvial Example:
----------------


vocabulary_list=['to',
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

CD_P=POS_CD(vocabulary_list)
CD_N=NEG_CD(vocabulary_list)
log_CD=LOG_CD(CD_P,CD_N)

###Writing to CSV file###
file_path=sys.argv[1]+'/log_CD.csv'
with open(file_path, 'w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(['word','log_CD(ti)'])
	
	for i in range(len(vocabulary_list)):
		wr.writerow([vocabulary_list[i],log_CD[i]])


print("log_CD for all the words in the vocabulary_list are as follows:-")
for i in range(len(vocabulary_list)):
	print(vocabulary_list[i],'------->',log_CD[i])
"""

		

