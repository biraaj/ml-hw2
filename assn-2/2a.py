# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:13:59 2023

@author: biraaj
"""

import numpy as np
from functions import load_data_svm
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#loading data
train_data_path= "./data/program_data_2c_2d_2e_train.txt"
test_data_path = "./data/program_data_2c_2d_2e_hw1_test.txt"

x_train,y_train = load_data_svm(train_data_path)

x_test,y_test = load_data_svm(test_data_path)

C = 6.0 #here we can change c values to get diffrent outputs for linear kernel
_svm = svm.SVC(kernel='linear', C=C)
_svm.fit(x_train, y_train)

y_pred_train = _svm.predict(x_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("training accuracy = ",accuracy_train)

y_pred_test = _svm.predict(x_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Testing accuracy = ",accuracy_test)

support_vectors = _svm.support_vectors_
print("support vectors length:",len(support_vectors))
print(support_vectors[0])
print(support_vectors[1])

#formatting array to get positive and negative index of target
_positive_index = y_train.ravel()==1
_negative_index = y_train.ravel()==-1


# Plotting 2D projections of data and decision boundary
fig, axis = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("Linear SVM decision boundary with c="+str(C))
titles = ['Height vs Diameter', 'Height vs Weight', 'Diameter vs Weight']

for i, proj in enumerate([(0, 1), (0, 2), (1, 2)]):
    #plot decision boundary
    w = _svm.coef_[0]
    a = -w[proj[0]]/w[proj[1]]
    xx = np.array([x_train[:,proj[0]].min(),x_train[:,proj[0]].max()])
    yy = a*xx - (_svm.intercept_[0])/w[proj[1]]
    axis[i].plot(xx,yy,"b-")

    # Plot data points.
    axis[i].plot(x_train[:,proj[0]][_positive_index], x_train[:,proj[1]][_positive_index], "go",label="positive(plastic)")
    axis[i].plot(x_train[:,proj[0]][_negative_index], x_train[:,proj[1]][_negative_index], "ro",label="negative(not plastic)")
    axis[i].set_title(titles[i])
    axis[i].legend(loc="lower right")

plt.savefig("./Results/linear_svm_with_c_"+str(C)+".jpeg")
plt.show()