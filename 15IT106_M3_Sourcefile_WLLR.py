import sys
import ast
from collections import Counter
from os import listdir
import simplejson
import math
import numpy as np
import csv
import pandas as pd



###Finding POS_WLLR of all the terms###
def POS_WLLR(vocabulary_list):
	pos_file_path=sys.argv[1]+'/pos_freq.csv'
	pos_df = pd.read_csv(pos_file_path)
	Number_Of_Positive_Documents=len(pos_df)

	neg_file_path=sys.argv[1]+'/neg_freq.csv'
	neg_df = pd.read_csv(neg_file_path)
	Number_Of_Negative_Documents=len(neg_df)

	D=Number_Of_Positive_Documents+Number_Of_Negative_Documents

	WLLR_P=[]
	for word in vocabulary_list:
		ncti = pos_df[word]
		ncti = ncti.tolist()
		ncti = np.count_nonzero(ncti)

		ncbti = neg_df[word]
		ncbti = ncbti.tolist()
		ncbti = np.count_nonzero(ncbti)

		first_part=float(ncti/Number_Of_Positive_Documents)
		if first_part == 0:
			WLLR_P_Per_Term=0
		else:
			numerator=ncti*(D - Number_Of_Positive_Documents)
			denominator=ncbti*Number_Of_Positive_Documents

			if denominator == 0 or float(numerator/denominator) == 0:
				WLLR_P_Per_Term=0
			else:
				WLLR_P_Per_Term=first_part*math.log(float(numerator/denominator),2)
		WLLR_P.append(WLLR_P_Per_Term)

	return WLLR_P


###Finding NEG_WLLR of all the terms###
def NEG_WLLR(vocabulary_list):
	pos_file_path=sys.argv[1]+'/pos_freq.csv'
	pos_df = pd.read_csv(pos_file_path)
	Number_Of_Positive_Documents=len(pos_df)

	neg_file_path=sys.argv[1]+'/neg_freq.csv'
	neg_df = pd.read_csv(neg_file_path)
	Number_Of_Negative_Documents=len(neg_df)

	D=Number_Of_Positive_Documents+Number_Of_Negative_Documents

	WLLR_N=[]
	for word in vocabulary_list:
		ncti = neg_df[word]
		ncti = ncti.tolist()
		ncti = np.count_nonzero(ncti)

		ncbti = pos_df[word]
		ncbti = ncbti.tolist()
		ncbti = np.count_nonzero(ncbti)

		first_part=float(ncti/Number_Of_Negative_Documents)
		if first_part == 0:
			WLLR_N_Per_Term=0
		else:
			numerator=ncti*(D - Number_Of_Negative_Documents)
			denominator=ncbti*Number_Of_Negative_Documents

			if denominator == 0 or float(numerator/denominator) == 0:
				WLLR_N_Per_Term=0
			else:
				WLLR_N_Per_Term=first_part*math.log(float(numerator/denominator),2)
		WLLR_N.append(WLLR_N_Per_Term)

	return WLLR_N


###Finding WLLR of all the terms###
def WLLR(WLLR_P,WLLR_N):
	WLLR_result=[]
	for i in range(len(WLLR_P)):
		WLLR_result.append(max(WLLR_P[i],WLLR_N[i]))

	return WLLR_result


###Accessing the Vocabulary###
file_path=sys.argv[1]+'/vocabulary.txt'
f1 = open(file_path, 'r')
vocabulary_list=f1.read()
vocabulary_list = ast.literal_eval(vocabulary_list)
f1.close()


###Performing WLLR###
WLLR_P=POS_WLLR(vocabulary_list)
WLLR_N=NEG_WLLR(vocabulary_list)
WLLR_result=WLLR(WLLR_P,WLLR_N)

###Writing to CSV file###
file_path=sys.argv[1]+'/WLLR.csv'
with open(file_path, 'w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(['word','WLLR(ti)'])
	
	for i in range(len(vocabulary_list)):
		wr.writerow([vocabulary_list[i],WLLR_result[i]])


print("WLLR for all the words in the vocabulary_list are as follows:-")
for i in range(len(vocabulary_list)):
	print(vocabulary_list[i],'------->',WLLR_result[i])



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

WLLR_P=POS_WLLR(vocabulary_list)
WLLR_N=NEG_WLLR(vocabulary_list)
WLLR_result=WLLR(WLLR_P,WLLR_N)


###Writing to CSV file###
file_path=sys.argv[1]+'/WLLR.csv'
with open(file_path, 'w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(['word','WLLR(ti)'])
	
	for i in range(len(vocabulary_list)):
		wr.writerow([vocabulary_list[i],WLLR_result[i]])


print("WLLR for all the words in the vocabulary_list are as follows:-")
for i in range(len(vocabulary_list)):
	print(vocabulary_list[i],'------->',WLLR_result[i])

"""






