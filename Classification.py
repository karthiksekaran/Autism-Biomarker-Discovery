from pandas import read_csv
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
array = read_csv("Final Reduced 26415 - Working-4.csv").values
X = array[:,:-1]
Y = array[:,-1]
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='eigen', store_covariance=False, tol=0.0001)
print(lda)
rf = RandomForestClassifier(n_estimators=10)
svm = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
clf = GridSearchCV(svm, parameters)
clf.fit(X,Y)
import numpy as np
print("accuracy:"+str(np.average(cross_val_score(clf, X, Y, scoring='accuracy'))))
print("f1:"+str(np.average(cross_val_score(clf, X, Y, scoring='f1'))))
lr = LogisticRegression()
nb = MultinomialNB()
knn = KNeighborsClassifier()
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results_lda = cross_val_predict(lda, X, Y, cv=kfold)
results_rf = cross_val_predict(rf, X, Y, cv=kfold)
results_svm = cross_val_predict(svm, X, Y, cv=kfold)
results_lr = cross_val_predict(lr, X, Y, cv=kfold)
results_nb = cross_val_predict(nb, X, Y, cv=kfold)
results_knn = cross_val_predict(knn, X, Y, cv=kfold)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
scores_lda = confusion_matrix(results_lda, Y)
scores_rf = confusion_matrix(results_rf, Y)
scores_svm = confusion_matrix(results_svm, Y)
scores_lr = confusion_matrix(results_lr, Y)
scores_nb = confusion_matrix(results_nb, Y)
scores_knn = confusion_matrix(results_knn, Y)
print("===============Linear Discriminant Analysis======================")
list = []
for i in scores_lda[0]:
    list.append(i)
for i in scores_lda[1]:
    list.append(i)
l = ["True Positive", "False Negative", "False Positive", "True Negative"]
print(l)
print(list)
print("Confusion Matrix:",scores_lda)
accuracy = accuracy_score(results_lda, Y)*100
print("Accuracy:",accuracy)
print("===============Random Forest======================")

list = []
for i in scores_rf[0]:
    list.append(i)
for i in scores_rf[1]:
    list.append(i)
l = ["True Positive", "False Negative", "False Positive", "True Negative"]
print(l)
print(list)
print("Confusion Matrix:",scores_rf)
accuracy = accuracy_score(results_rf, Y)*100
print("Accuracy:",accuracy)
print("===============Support Vector Machines======================")

list = []
for i in scores_svm[0]:
    list.append(i)
for i in scores_svm[1]:
    list.append(i)
l = ["True Positive", "False Negative", "False Positive", "True Negative"]
print(l)
print(list)
print("Confusion Matrix:",scores_svm)
accuracy = accuracy_score(results_svm, Y)*100
print("Accuracy:",accuracy)

print("===============Logistic Regression======================")

list = []
for i in scores_lr[0]:
    list.append(i)
for i in scores_lr[1]:
    list.append(i)
l = ["True Positive", "False Negative", "False Positive", "True Negative"]
print(l)
print(list)
print("Confusion Matrix:",scores_lr)
accuracy = accuracy_score(results_lr, Y)*100
print("Accuracy:",accuracy)

print("===============Naive Bayes======================")

list = []
for i in scores_nb[0]:
    list.append(i)
for i in scores_nb[1]:
    list.append(i)
l = ["True Positive", "False Negative", "False Positive", "True Negative"]
print(l)
print(list)
print("Confusion Matrix:",scores_nb)
accuracy = accuracy_score(results_nb, Y)*100
print("Accuracy:",accuracy)

print("===============K-Nearest Neighbor======================")

list = []
for i in scores_knn[0]:
    list.append(i)
for i in scores_knn[1]:
    list.append(i)
l = ["True Positive", "False Negative", "False Positive", "True Negative"]
print(l)
print(list)
print("Confusion Matrix:",scores_knn)
accuracy = accuracy_score(results_knn, Y)*100
print("Accuracy:",accuracy)
#ROC Curve

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
fpr, tpr, thresholds = roc_curve(results_lda, Y)
roc_auc = auc(fpr, tpr)
lw=2
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
