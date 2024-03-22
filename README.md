# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:kabilan V 
RegisterNumber: 212222100018
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```


## Output:
## dataset
![image](https://github.com/kabilan22000284/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123469171/35e77c72-470f-43c2-bfe1-72514be12594)

## headvalue
![image](https://github.com/kabilan22000284/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123469171/9052fc40-6d2f-4dfc-ac37-13e75e7eb36c)

## tailvalue
![image](https://github.com/kabilan22000284/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123469171/20844141-d694-4d48-b45b-7bee3f657464)

## X and Y values
![image](https://github.com/kabilan22000284/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123469171/831d9764-d4f7-4b28-b96b-9099d895c544)

## predicted X and Y value
![image](https://github.com/kabilan22000284/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123469171/59704ba1-05a8-460d-92b3-ff590e3e0713)

## MSE,MAE and MRAE
![image](https://github.com/kabilan22000284/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123469171/e4cd2e9b-e71d-41e6-8d7a-c76a9f64473d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
