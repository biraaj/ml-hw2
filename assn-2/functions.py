# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:14:12 2023

@author: biraaj
"""

import numpy as np
import pandas as pd


def replace_parenthesis(_string):
    """
        Function to filter out data and split by removing parenthesis.
        Note: This implmentation was from hw1
    """
    return _string.replace('(','').replace(')','').replace(' ','').strip().split(",")

def load_data_svm(train_data_path):
    """
        This is a function to load the training data for SVM classification problems.
    """
    _feat = []
    _target = []
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append([float(i) for i in temp_data[:(len(temp_data)-1)]])
            if(temp_data[len(temp_data)-1] == "Plastic"):
                _target.append(1)
            else:
                _target.append(-1)
    
    return np.array(_feat),np.array(_target)


# Decision Tree Implementation

def load_data_dt(train_data_path):
    """
        This is a function to load the training data for decision tree classification problems.
    """
    _feat = []
    _target = []
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthesis(_row)
            # mapping is done below to convert floating points and string to their respective format from a all string text.
            _feat.append([float(i) for i in temp_data[:(len(temp_data)-1)]])
            _target.append(temp_data[-1])
    
    return np.array(_feat),np.array(_target)




class DecisionTree:
    def __init__(self, depth=None):
        self.maximum_depth = depth
        
    def fit(self, feature, target):
        self.total_classes = len(np.unique(target))
        self.decison_tree = self.tree_build(feature, target)
        
    def _predict(self, _feature, d_tree):
        if isinstance(d_tree, tuple):
            _feat, _thresh, left_tree, right_tree = d_tree
            if _feature[_feat] <= _thresh:
                return self._predict(_feature, left_tree)
            else:
                return self._predict(_feature, right_tree)
        else:
            return d_tree
        
    def calculate_entropy(self, target):
        ent = 0
        for _class in range(self.total_classes):
            probab_c = len(target[target == _class]) / len(target)
            if probab_c > 0:
                ent -= probab_c * np.log2(probab_c)
        return ent
        
    def tree_build(self, feature, target, depth=0):
        sample_count, feature_count =  feature.shape
        total_classes = len(np.unique(target))
        
        # edge cases to stop going deep
        if depth == self.maximum_depth or total_classes == 1 or sample_count < 2:
            return self.leaf(target)
        
        # using information gain to split the tree
        maximum_gain = -np.inf
        for _index in range(feature_count):
            for _val in np.unique(feature[:, _index]):
                right_node = feature[:, _index] > _val
                left_node = feature[:, _index] <= _val
                
                if sum(left_node) == 0 or sum(right_node) == 0:
                    continue
                information_gain = self._info_gain(target, [target[left_node], target[right_node]])
                
                if information_gain > maximum_gain:
                    maximum_gain = information_gain
                    _feat_left, _targ_left = feature[left_node], target[left_node]
                    _feat_right, _targ_right = feature[right_node], target[right_node]
                    best_feat = _index
                    optimal_value = _val
                    
        # build subtrees
        if information_gain > 0:
            left_sub_tree = self.tree_build(_feat_left, _targ_left, depth+1)
            right_sub_tree = self.tree_build(_feat_right, _targ_right, depth+1)
            return (best_feat, optimal_value, left_sub_tree, right_sub_tree)
        
        # End condition when there is no more gain
        return self.leaf(target)
    
    def leaf(self, target):
        return np.bincount(target).argmax()
    
    def _info_gain(self, _parent, _childs):
        previous_ent = self.calculate_entropy(_parent)
        later_ent = 0
        
        for child in _childs:
            if len(child) > 0:
                later_ent += len(child) / len(_parent) * self.calculate_entropy(child)
        
        return previous_ent - later_ent
    
    
    def return_predictions(self, feature):
        return [self._predict(_feat, self.decison_tree) for _feat in feature]

def compute_accuracy(pred,actual):
    correct_output = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct_output += 1
    return correct_output/len(actual)


