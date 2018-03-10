import sys
from os import listdir
from nltk import PorterStemmer 
import nltk 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
import string
from collections import Counter
import math 
import numpy as np 
import simplejson 
import ast 


def loading_all_files(directory_path):
	for file in listdir(directory_path):
		file_path=directory_path+"/"+file
		load_a_file(file_path,directory_path,file)

def load_a_file(file_path,directory_path,file_name):
	file=open(file_path,'r')
	data=file.read()
	tokenization(data,directory_path,file_name)
	file.close()

###Step1 of Preprocessing---Tokenization###
def tokenization(data,directory_path,file_name):
	token_list=data.split()
	split_directory_path=directory_path.split("/")
	file_path=sys.argv[1]+'/Tokenization/'+split_directory_path[-1]+"/"+file_name
	f = open(file_path, 'w')
	simplejson.dump(token_list, f)
	f.close()
	stemming(token_list,directory_path,file_name)


###Step2 of Preprocessing---Normalization(It can be assumed normalized)###

###Step3 of Preprocessing---Stemming###
def stemming(token_list,directory_path,file_name):
	split_directory_path=directory_path.split("/")
	file_path=sys.argv[1]+'/Stemming/'+split_directory_path[-1]+"/"+file_name
	for i in range(len(token_list)):
		stemming_token=PorterStemmer().stem(token_list[i])
		token_list[i]=stemming_token
	f = open(file_path, 'w')
	simplejson.dump(token_list, f)
	f.close()
	cleaning_and_stopword_removal(token_list,directory_path,file_name)


###Step4 of Preprocessing---Cleaning & Stopword Removal###
'''
Cleaning should have the following steps:
(i) Removing tokens that contains punctuations
(ii) Removing tokens that are just punctuations
(iii) Removing tokens that contain numbers
(iv) Removing tokens that have one character 
'''
def cleaning_and_stopword_removal(token_list,directory_path,file_name):
	split_directory_path=directory_path.split("/")
	file_path=sys.argv[1]+'/Cleaning_Stopword_Removal/'+split_directory_path[-1]+"/"+file_name
	###(i) & (ii)###
	translation = str.maketrans("","", string.punctuation);
	for i in range(len(token_list)):
		token_list[i]=token_list[i].translate(translation)

	###(iii)###
	token_list=[token for token in token_list if token.isalpha()]

	###(iv)###
	token_list=[token for token in token_list if len(token)>1]

	###Stopword Removal###
	stop_words = set(stopwords.words('english'))
	token_list=[token for token in token_list if not token in stop_words]

	f = open(file_path, 'w')
	simplejson.dump(token_list, f)
	f.close()



###Creating a vocabulary###
def create_vocabulary(pos_directory_path,neg_directory_path):
	vocabulary=Counter()
	for file in listdir(pos_directory_path):
		file_path=pos_directory_path+"/"+file
		f=open(file_path,'r')
		data=f.read()
		data = ast.literal_eval(data)
		vocabulary.update(data)
		f.close()

	for file in listdir(neg_directory_path):
		file_path=neg_directory_path+"/"+file
		f=open(file_path,'r')
		data=f.read()
		data = ast.literal_eval(data)
		vocabulary.update(data)
		f.close()
	vocabulary_list = [word for word,frequency in vocabulary.items() if frequency >= 5]
	file_path=sys.argv[1]+'/vocabulary.txt'
	f = open(file_path, 'w')
	simplejson.dump(vocabulary_list, f)
	f.close()



directory_path_for_positive_files="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/"+sys.argv[1]+"/pos"
#directory_path_for_positive_files=['Absolute_Path_To_This_Directory']+sys.argv[1]+"/pos"
directory_path_for_negative_files="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/"+sys.argv[1]+"/neg"
#directory_path_for_negative_files=['Absolute_Path_To_This_Directory']+sys.argv[1]+"/neg"
loading_all_files(directory_path_for_positive_files)
loading_all_files(directory_path_for_negative_files)

###For Creating the vocabulary###
directory_path_for_positive_files="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/"+sys.argv[1]+"/Cleaning_Stopword_Removal/pos"
#directory_path_for_positive_files=['Absolute_Path_To_This_Directory']+sys.argv[1]+"/Cleaning_Stopword_Removal/pos"
directory_path_for_negative_files="/Users/aimanabdullahanees/Desktop/Sentiment_Analysis/"+sys.argv[1]+"/Cleaning_Stopword_Removal/neg"
#directory_path_for_negative_files=['Absolute_Path_To_This_Directory']+sys.argv[1]+"/Cleaning_Stopword_Removal/neg"
create_vocabulary(directory_path_for_positive_files,directory_path_for_negative_files)

print("Preprocessing Successful!")




