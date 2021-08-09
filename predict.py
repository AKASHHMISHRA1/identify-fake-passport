#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:25:12 2020

@author: akash
"""
import numpy as np
import csv
import sys
import pickle
from validate import validate

def import_dataset(train_X_filepath,train_Y_filepath):
    train_X = np.genfromtxt(train_X_filepath, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_filepath, delimiter=',', dtype=np.int64, skip_header=0)
    
    return train_X,train_Y


def calculate_gini_index(Y_subsets):
    #TODO Complete the function implementation. Read the Question text for details
    gini=0
    no_of_elements=0
    #print(Y_subsets)
    for x in Y_subsets:
        count=(np.unique(x,return_counts=True)[1])
        probability=count/np.sum(count)
        gini+=(1-np.sum(np.dot(probability,probability)))*len(x)
        no_of_elements+=len(x)
    return gini/no_of_elements


def split_data_set(data_X, data_Y, feature_index, threshold):
    #TODO Complete the function implementation. Read the Question text for details
    data_X=np.array(data_X)
    data_Y=np.array(data_Y)
    check_threshold=data_X.T[feature_index]
    row_index=check_threshold<threshold
    row_greater=check_threshold>=threshold
    return data_X[row_index],data_Y[row_index],data_X[row_greater],data_Y[row_greater]


def get_best_split(X, Y):
    #TODO Complete the function implementation. Read the Question text for details
    best_feature=9999
    best_threshold=9999
    best_gini=9999
    for x in range(len(X[0])):
       for y in range(len(X)): 
         left_X,left_Y,right_X,right_Y=split_data_set(X,Y,x,X[y][x])
         #print(left_X, left_Y, right_X, right_Y)
    
         gini=calculate_gini_index(np.array([left_Y,right_Y]).tolist())
         #print(x,y,gini)
         if gini<best_gini:
             best_gini=gini
             best_feature=x
             best_threshold=X[y][x]
         elif gini==best_gini and best_feature==x :
             if best_threshold>X[y][x]:
               #print(X[y][x])
               best_threshold=X[y][x]
    return int(best_feature),best_threshold

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

def construct_tree(X, Y, max_depth, min_size, depth):
    Y = np.array(Y)
    classes = list(set(Y))
    #print(classes)
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)

    #check is pure
    if len(set(Y)) == 1:
        return node
    
    #check max depth reached
    if depth >= max_depth:
        return node

    #check min subset at node
    if len(Y) <= min_size:
        return node

    feature_index, threshold = get_best_split(X, Y)

    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth + 1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth + 1)
    return node

def print_tree(node):
    if node.left is not None and node.right is not None:
        print("X" + str(node.feature_index) + " " + str(node.threshold))
    if node.left is not None:
        print_tree(node.left)
    if node.right is not None:
        print_tree(node.right)

def predict(root, X):
    #TODO Complete the function implementation. Read the Question text for details
    #print(root.predicted_class,root.feature_index,root.threshold,root.depth,root.left,root.right)
    if X[root.feature_index]<root.threshold and root.left is not None:
     return predict(root.left,X)
    elif root.left is None:
     #print(root.predicted_class)
     return root.predicted_class
    if X[root.feature_index]>=root.threshold and root.right is not None:
     return predict(root.right,X)
    elif root.right is None:
     #print(root.predicted_class)
     return root.predicted_class
 

def train_data(test_X_file_path,train_X_filepath,train_Y_filepath):
    train_X_data,train_Y_data=import_dataset(train_X_filepath,train_Y_filepath)
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    #print(train_X_data,train_Y_data)
    root=construct_tree(train_X_data,train_Y_data,10,1,0)
    #print(root)
    predicted_values=[]
    #print(len(predicted_values))
    #print(test_X)
    #print(predicted())
    print(train_Y_data)
    print(set(train_Y_data))
    for X in test_X:
       predicted_values.append(predict(root,X))
    #print(predicted_values)
    write_to_csv_file(np.array(predicted_values),"predicted_test_Y_de.csv")



def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
    
    
    
if __name__ == "__main__":
    test_X_file_path =sys.argv[1]
    train_data(test_X_file_path,'train_X_de.csv','train_Y_de.csv')
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 
