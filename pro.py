
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC

dataset=pd.read_csv('mo.csv')

x = dataset.drop('class',1)
y = dataset['class']


##spliting data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)




## preprocessing feature scaler


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## preprocessing PCA
##decrease diminsions

pca=PCA(n_components=2)

X_train=pca.fit_transform(X_train)

X_test=pca.fit_transform(X_test)

'''''
print(X_train.shape)
print(pca_x_train.components_.shape)
'''''


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


## RandomForestClassifier

print('RandomForestClassifier')

model = RandomForestClassifier()
model.fit(X_train, y_train)
predict = model.predict(X_test)

acc = accuracy_score(y_test,predict)
print(acc)


conf = confusion_matrix(y_test,predict)
print(conf)

print(classification_report(y_test,predict))

print(' ************************************** /n ')


## Artificial_Neural_Networks

print('Artificial_Neural_Networks')
##500  itration max to calc waight

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

mlp.fit(X_train,y_train)

y_pred_1 = mlp.predict(X_test)

acc1=accuracy_score(y_test,y_pred_1)
print(acc1)

cm_1 = confusion_matrix(y_test, y_pred_1)
print(cm_1)

print(classification_report(y_test,y_pred_1))

print(' ************************************** /n ')



##knn
##point neighbors the best naig  be 1,3,5

print('k neighbors classifier')

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred_2 = classifier.predict(X_test)

acc2 =accuracy_score(y_test,y_pred_2)

print(acc2)

print(confusion_matrix(y_test, y_pred_2))

print(classification_report(y_test, y_pred_2))

print(' ************************************** /n ')



##suport vector machine
##line or rpf to classifier data >>>

print('suport vector machine')

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred_3 = svclassifier.predict(X_test)
acc3 =accuracy_score(y_test,y_pred_3)
print(acc3)
cm_3 = confusion_matrix(y_test, y_pred_3)
print(cm_3)

print(classification_report(y_test,y_pred_3))

print(' ************************************** /n ')



### comparing models

print('best model with highest accuracy')
print(max(acc,acc1,acc2,acc3))

