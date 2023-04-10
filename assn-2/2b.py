# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:12:20 2023

@author: biraaj
"""

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

#loading data
train_data_path= "./data/program_data_2c_2d_2e_train.txt"
test_data_path = "./data/program_data_2c_2d_2e_hw1_test.txt"

x_train,y_train = load_data_svm(train_data_path)

x_test,y_test = load_data_svm(test_data_path)

_C = 6.0
_gamma = 5.0
_svm = svm.SVC(kernel='rbf', C=_C, gamma=_gamma)
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
print(support_vectors[2])

#formatting array to get positive and negative index of target
_positive_index = y_train.ravel()==1
_negative_index = y_train.ravel()==-1

# Plot 2D projections of data and decision boundary
fig, axis = plt.subplots(1, 3, figsize=(20, 5))
#plt.title("Linear SVM decision boundary with c=",C)
fig.suptitle("Gaussian SVM decision boundary with c="+str(_C)+"and gamma = "+str(_gamma))
titles = ['Height vs Diameter', 'Height vs Weight', 'Diameter vs Weight']

#referred from https://gtraskas.github.io/post/ex6/
plot_1 = np.linspace(x_train[:,0].min(), x_train[:,0].max(), 100).T
plot_2 = np.linspace(x_train[:,1].min(), x_train[:,1].max(), 100).T
plot_3 = np.linspace(x_train[:,2].min(), x_train[:,2].max(), 100).T
plot_4 = np.linspace(x_train[:,3].min(), x_train[:,3].max(), 100).T
p1, p2 = np.meshgrid(plot_1, plot_2)
p3, p4 = np.meshgrid(plot_3, plot_4)
X_list =[p1,p2,p3,p4] 
value = np.zeros(p1.shape)
for i in range(p1.shape[1]):
    this_X = np.column_stack((p1[:, i], p2[:, i],p3[:,i],p4[:,i]))
    value[:, i] = _svm.predict(this_X)

for i, proj in enumerate([(0, 1), (0, 2), (1, 2)]):
    #plot decision boundary
    axis[i].contour(X_list[proj[0]], X_list[proj[1]], value, colors="b",levels=[-1,0,1])

    # Plot data points.
    axis[i].plot(x_train[:,proj[0]][_positive_index], x_train[:,proj[1]][_positive_index], "go",label="positive(plastic)")
    axis[i].plot(x_train[:,proj[0]][_negative_index], x_train[:,proj[1]][_negative_index], "ro",label="negative(not plastic)")
    axis[i].set_title(titles[i])
    axis[i].legend(loc="lower right")
    
plt.savefig("./Results/Gaussian_svm_with_c_"+str(_C)+"_gamma_"+str(_gamma)+".jpeg")
plt.show()