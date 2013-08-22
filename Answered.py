'''
Contest : QUORA ML CODESPRINT 2013 
Problem : QUORA ML PROBLEM: ANSWERED
		  For this task, given Quora question text and topic data, 
		  predict whether a question gets an upvoted answer within 1 day.
Username: gshguru
Date    : 28th July 2013 (FINAL VERSION)
'''
''' for reading and formatting input '''
import json

''' for Text Preprocessing '''
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=False)

''' for training '''
from sklearn.linear_model import LogisticRegression

''' variables aka data stores '''
train_data = []			# data store for training data, list of strings
test_data = []			# data store for test data, list of strings
train_y = []			# data store for target variables for training, list of results
train_Qid = []			# list Question Ids for train_data
test_Qid = []			# list Question Ids for test_data

''' feature Words are words extracted from top features mined using SGD classifier '''
featureWords = ['mongodb','throat','pakistan','memorization',
				'slow','independent','designer','barre'] #icpc was there in top 25!

''' convert input json data to the appropriate lists and data stores '''
def convert(x, mode = 'train'):
	ob = json.loads(x)
	text = ob['question_text']
	text += ", "+str(ob['anonymous'])+"an"  # adding a feature from anonymous tag
	for k, v in ob.items():					# topics		
		if type(v) == list:			
			for topic in v:				
				text += " "+topic['name']	# add topics to string
				# as number of followers also a significance/relevance factor,
				# adding one feature realted to follwers 
				if(topic['followers'] > 1100): text += " "+'featureAddon'

		elif type(v) == dict:				# context topic
			name = v['name']
			followers = v['followers']
			text += " "+name				# updating text with context topic
			if (followers > 550): text += " "+'featureAddon'	

	if (mode == 'test'): return text, ob['question_key']
	for word in featureWords:
		if(word in text): text += "featureAddonfw" #adding one extra feature using SGD
	return text, ob['__ans__'], ob['question_key']

def predict():
	# training set text processing
	X_train_counts = count_vect.fit_transform(train_data)	
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	# training set text processing
	X_new_counts = count_vect.transform(test_data)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	# initializing model
	clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.5
							, fit_intercept=True, intercept_scaling=10, class_weight='auto'
							, random_state=None)

	# model fitting
	clf = clf.fit(X_train_tfidf, train_y)

	# predicting answers for test set
	predicted = clf.predict(X_new_tfidf)	

	# writing to STDOUT in JSON format
	for i in range(len(predicted)):		
		tempDict = {}
		tempDict['__ans__'] = bool(predicted[i]) #numpy element to bool
		tempDict['question_key'] = test_Qid[i]
		print json.dumps(tempDict)
	
''' reading training data from STDIN '''
N = int(raw_input())
for i in range(N):
	line = raw_input()
	txt, ans, q = convert(line)
	train_Qid.append(q)
	train_data.append(txt)
	train_y.append(ans)

''' reading test data from STDIN '''
T = int(raw_input())
for i in range(T):
	line = raw_input()
	txt, q = convert(line, 'test')
	test_data.append(txt)
	test_Qid.append(q)

''' build model, predict and print results '''
predict()
