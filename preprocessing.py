from os import listdir
from nltk import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from collections import Counter

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
print(len(documents))
print(len(vocabulary_list))

