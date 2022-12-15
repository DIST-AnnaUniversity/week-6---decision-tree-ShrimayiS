import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.preprocessing import LabelEncoder 



df = pd.read_csv("C:\\Users\\Admin\\Desktop\\ex7\\emp.csv")
df.dropna(inplace=True)
X=df.drop(["Attrition" , "EmployeeNumber" , "Over18"], axis=1)
y=df["Attrition"]


X_train, X_test, y_train ,y_test= train_test_split(X,y,test_size=0.25,random_state=30)
lc=LabelEncoder()
tf=lc.fit_transform(X_train['BusinessTravel'])
X_train['BusinessTravel']=tf
tf=lc.transform(X_test['BusinessTravel'])
X_test['BusinessTravel']=tf
tf=lc.fit_transform(X_train['Department'])
X_train['Department']=tf
tf=lc.transform(X_test['Department'])
X_test['Department']=tf
tf=lc.fit_transform(X_train['EducationField'])
X_train['EducationField']=tf
tf=lc.transform(X_test['EducationField'])
X_test['EducationField']=tf
tf=lc.fit_transform(X_train['Gender'])
X_train['Gender']=tf
tf=lc.transform(X_test['Gender'])
X_test['Gender']=tf
tf=lc.fit_transform(X_train['JobRole'])
X_train['JobRole']=tf
tf=lc.transform(X_test['JobRole'])
X_test['JobRole']=tf
tf=lc.fit_transform(X_train['MaritalStatus'])
X_train['MaritalStatus']=tf
tf=lc.transform(X_test['MaritalStatus'])
X_test['MaritalStatus']=tf
tf=lc.fit_transform(X_train['OverTime'])
X_train['OverTime']=tf
tf=lc.transform(X_test['OverTime'])
X_test['OverTime']=tf

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

plt.figure(figsize=(15,15))

tree.plot_tree(dtree) 
plt.savefig("tree.png")