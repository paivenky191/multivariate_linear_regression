# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:02:16 2018

@author: Dell
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from math import sqrt


dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [46,62]].values
Y = dataset.iloc[:, 80].values

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

price=Y_train[:]
area=X_train[:, 0]
g_area=X_train[:,1]

#x1=area
#x2=g_area
#y1=price


#plots
#sns.distplot(np.log1p(price))


#get correlation
def getCorr(x,y):
    return np.corrcoef(x,y)[0][1]

#get regression coefficient
def get_reg_coeff(x1,x2,y):
    right_side = y.std() / x1.std()
    y_x1_corr=getCorr(y,x1)
    y_x2_corr=getCorr(y,x2)
    x1_x2_corr=getCorr(x1,x2)
    num=y_x1_corr-(y_x2_corr * x1_x2_corr)
    denom=1-(x1_x2_corr**2)
    left_side=num/denom
    reg_coeff=left_side * right_side
    return reg_coeff


#coefficients(regression)

b1=get_reg_coeff(area,g_area,price)
b2=get_reg_coeff(g_area,area,price)

def get_y_intercept(x1,x2,y):
    b1=get_reg_coeff(x1,x2,y)
    b2=get_reg_coeff(x2,x1,y)
    a= y.mean()-(b1*x1.mean())-(b2*x2.mean())
    return a

y_intercept=get_y_intercept(area,g_area,price)


def predict_price(new_area,new_g_area):
    return((new_area*b1)+(new_g_area*b2)+y_intercept)


my_prediction=[]
#my_price=(my_area*b1)+(my_g_area*b2)+y_intercept
for data in X_test:
    pred_area=data[0]
    pred_g_area=data[1]
    predicted_price=predict_price(pred_area,pred_g_area)
    my_prediction.append(predicted_price)

my_error=[]
sum=0
n=len(Y_test)
for i in range(0,len(Y_test)):
    print(n)
    val=(Y_test[i]-my_prediction[i])
    #sum+=(Y_test[i]-my_prediction[i])**2
    #mse=sum/n
    my_error.append(val)
plt.plot(Y_test)
plt.plot(my_prediction)
    

