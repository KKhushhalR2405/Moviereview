import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB


def cleandata(file):
	file.dropna(inplace=True)
	blank=[]
	for i in file.itertuples():
		if i[2].isspace():
			blank.append(i[0])
	file.drop(blank,inplace=True)



file=pd.read_csv("D:\\Projects\\Movie Review\\review.tsv",sep='\t')


#print(file.isnull().sum())

cleandata(file)

#print(file.isnull().sum())

X=file['review']
y=file['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

#text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB())])

text_clf.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

p=text_clf.predict(X_test)

print(confusion_matrix(y_test,p))

print(classification_report(y_test,p))

print(int(accuracy_score(y_test,p)*100))

#If you wanna check your model :  
"""p=text_clf.predict(["I think its a waste of time if you see this"])

print(p)"""


