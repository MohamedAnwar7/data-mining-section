import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
data =pd.read_csv('diabetics.csv')
x= data.drop('outcome',1)
y = data('outcome')
print(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = RandomForestClassifier()
model.fit(x_train,y_train)
predict = model.predict(x_test)
acc = accuracy_score(predict,y_test)
print(acc)
conf = confusion_matrix(predict,y_test)
print(conf)
