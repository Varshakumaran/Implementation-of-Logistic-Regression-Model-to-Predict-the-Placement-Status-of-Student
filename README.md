# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection & Preprocessing

2.Select relevant features that impact placement

3.Import the Logistic Regression model from sklearn.

4.Train the model using the training dataset.

5.Use the trained model to predict placement for new student data. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VARSHA K
RegisterNumber:  212223220122
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
## Output:

![image](https://github.com/user-attachments/assets/aba2c347-59dc-4869-ba02-164210623a10)


```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

## OUTPUT: 
![image](https://github.com/user-attachments/assets/8437fab4-ae53-4465-bc4a-d7a24e3d6884)

```
data1.isnull().sum()

```
## OUTPUT:
![image](https://github.com/user-attachments/assets/d838ae4a-8bae-4b18-b4a0-81e2b2b6f73f)

```
data1.duplicated().sum()
```
## OUTPUT:
 ![image](https://github.com/user-attachments/assets/ed46a811-16ff-4807-a5da-5ea45492c593)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/f2030333-1553-4d1c-95d9-80096a1f49fa)

```
x=data1.iloc[:,:-1]
x
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/6a7b2e9c-b56c-4fc3-9fc7-b727631da3f0)

```
y=data1["status"]
y
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/7f47dd60-2172-4cf9-85a2-c6133febc6ba)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```

```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
## OUTPUT:

![image](https://github.com/user-attachments/assets/166d7680-6d0d-40e8-a712-a4617811f2ee)
```

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/ee210ec4-59e0-45b0-bb18-865f5518d0dc)

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/f06ebafe-01b2-40e0-8779-785e689dcaac)

```

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## OUTPUT:

![image](https://github.com/user-attachments/assets/83827e19-b0f6-4537-8345-cee5f67ff0b7)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
