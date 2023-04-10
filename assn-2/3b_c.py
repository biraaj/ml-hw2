# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:41:48 2023

@author: biraaj
"""

import numpy as np
from functions import DecisionTree,load_data_dt,compute_accuracy

data_path = "./data/program_data_2c_2d_2e_hw1.txt"

x_train,y_train = load_data_dt(data_path)

# Define class labels and map them to integers
class_labels = ['Plastic','Metal','Ceramic']
label_map = {label: idx for idx, label in enumerate(class_labels)}
y_train_modified = np.array([label_map[label] for label in y_train])

print("answer for question 3b:")
print("class labels:",class_labels)
depth=12
for i in range(1,depth):
    # Train decision tree
    tree = DecisionTree(depth=i)
    tree.fit(x_train, y_train_modified)
    
    # Test decision tree
    y_pred = tree.return_predictions(x_train)
    print("training accuracy for the decision tree with max depth = ",str(i)," is: ",compute_accuracy(y_pred,y_train_modified))
    
print("........................................................................................................")
print()

print("answer for question 3c:")
print("class labels:",class_labels)
train_data_path= "./data/program_data_2c_2d_2e_train.txt"
test_data_path = "./data/program_data_2c_2d_2e_hw1_test.txt"
x_train,y_train = load_data_dt(train_data_path)
x_test,y_test = load_data_dt(test_data_path)
y_train_modified = np.array([label_map[label] for label in y_train])
y_test_modified = np.array([label_map[label] for label in y_test])

depth = 9
for i in range(1,depth):
    # Train decision tree
    tree = DecisionTree(depth=i)
    tree.fit(x_train, y_train_modified)
    
    # Test decision tree
    y_pred_train = tree.return_predictions(x_train)
    print("training accuracy for the decision tree with max depth = ",str(i)," is: ",compute_accuracy(y_pred_train,y_train_modified))
    
    y_pred_test = tree.return_predictions(x_test)
    print("testing accuracy for the decision tree with max depth = ",str(i)," is: ",compute_accuracy(y_pred_test,y_test_modified))
    print("***********")

