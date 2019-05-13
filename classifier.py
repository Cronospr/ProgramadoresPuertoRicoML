import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

#dataset
df = pd.read_csv("dataset.csv")

#normalization
def clean_str(string):
	string = re.sub(r"\n", "", string)
	string = re.sub(r"\r", "", string)
	string = re.sub(r"[0-9]", "digit", string)
	string = re.sub(r"\'", "", string)
	string = re.sub(r"\"", "", string)
	return string.strip().lower()


X = []
for i in range(df.shape[0]):
	X.append(clean_str(df.iloc[i][1]))
y = np.array(df["category"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = Pipeline([('vectorizer', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
			  'tfidf__use_idf': (True, False)}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X, y)
#saving accuracy in file
f = open("accuracy.txt", "w+")
print >>f, gs_clf_svm.best_score_
f.close()
model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
				   ('tfidf', TfidfTransformer(use_idf=True)),
				   ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
#saving classifiers in file
f = open("classifiers.txt", "w+")
print >>f, model.classes_
f.close()
np.set_printoptions(threshold=np.inf)
#saving confussion matrix in file
f = open("confMatrix.txt", "w+")
print >>f, confusion_matrix(pred, y_test)
f.close()
#print test categories
f = open("testCategories.txt", "w+")
print >>f, y_test
f.close()
