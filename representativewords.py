# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:09:11 2016

@author: Ying Zhou
"""

import numpy as np
import math 
import string as strings

##### read file #####
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
pprint(list(newsgroups_train.target_names))
newsgroups_test = fetch_20newsgroups(subset='test')

X_train = newsgroups_train.data
y_train = newsgroups_train.target

X_test = newsgroups_test.data
y_test = newsgroups_test.target

with open('vocabulary.txt') as f:
    V = f.readlines()

# transfer V to dictionary
V = dict(enumerate(V))
V = {y:x for x,y in V.items()}

##### calculate the fraction of documents that belong to that class #####
pi_frac = []
for i in range(20):
    freq = (y_train == i).sum()
    pi_frac.append(freq/len(y_train))


##### calculate P_j #####
# find all the documents of class j and combine them, put them in list classes 
classes = [None]*20
for j in range(20):
    index = list(np.where(y_train == j))
    result = []
    for i in index[0]:
        string = X_train[i]
        for char in string:
            if char in strings.punctuation:
                string = string.replace(char, ' ')
                string = string.lower()
        result = result + string.split()
    classes[j] = result

# calculate count of a word in classes, return a 20*len(V) shape matrix 
def counts(classes, V):
    matrix = np.zeros([20,len(V)]) 
    for i in range(20):
        for word in classes[i]:
            value = V.get(word+'\n')
            if value != None:
                matrix[i][value] = matrix[i][value] + 1
    return matrix

# calculate x, given V and document x
def x_occurance(x, V):
    result = np.zeros(len(V))
    for char in x:
            if char in strings.punctuation:
                x = x.replace(char, ' ')
    x = x.lower()
    string = x.split()
    for word in string:
        value = V.get(word+'\n')
        if value != None:
            result[value] = result[value] + 1
    return result

##### classify document x #####
# x is test document, V is vocabulary list, matrix is 20*len(V) p_ji matrix
def classify(x, V, matrix):
    max = -10000
    index = 0
    for j in range(20):
        tmp_max = math.log(pi_frac[j]) # initial value of the formula
        x_occur = x_occurance(x, V)
        sum_j = matrix[j].sum()
        for i in range(len(V)):
            tmp_max = tmp_max + x_occur[i]*math.log((matrix[j][i] + 0.1)/(sum_j+ 0.1*len(V))) # smoothing
        if tmp_max > max:
            max = tmp_max
            index = j
    return index
    

##### Error rate of test documents #####
def error_rate(X_test, y_test, V):
    matrix = counts(classes, V)
    test_result = []
    accur = 0
    for x in X_test:
        class_idx = classify(x, V, matrix)
        test_result.append(class_idx)
    print('done')
    y_test = list(y_test)
    for i in range(len(y_test)):
        if y_test[i] == test_result[i]:
            accur = accur + 1
    return 1- accur/len(y_test)

# random select #
import random

def random_error(X_test, y_test, V, N):
    rand_idx = random.sample(range(len(V)), N)
    rand_V = []
    for idx in rand_idx:
        rand_V.append(V[idx])
    rand_V = dict(enumerate(rand_V))
    rand_V = {y:x for x,y in rand_V.items()}
    error = error_rate(X_test, y_test, rand_V)
    return error

##### chose N vocabulary #####
# here, V is list, should be transfered to dictionary when used in error_rate
def choose_voc(V, N):
    dict_V = dict(enumerate(V))
    dict_V = {y:x for x,y in dict_V.items()}
    matrix = counts(classes, dict_V)
    chosen_V = []
    means = []
    maxes = []
    for i in range(len(V)):
        means.append(matrix[:,i].mean())
    for i in range(len(V)):
        maxes.append(matrix[:,i].max())
    diff = np.zeros(len(V))
    for i in range(len(V)):
        diff[i] = maxes[i] - means[i]
    indexes = np.argsort(diff)[-N:]
    for index in indexes:
        chosen_V.append(V[index])
    chosen_V = dict(enumerate(chosen_V))
    chosen_V = {y:x for x,y in chosen_V.items()}
    return chosen_V

for N in [100,500, 1000, 5000, 10000, 20000]:
    print(N)
    print('random ')
    print(random_error(X_test, y_test, V, N))
    print('chosen')
    chosen_V = choose_voc(V, N)
    print(error_rate(X_test, y_test, chosen_V))

def counts_2(diff):
    vocs = []
    result = []
    for i in range(20):
        index = list(np.where(y_train == i))
        local = []
        for idx in index:
            local.append(diff[idx])
        index = np.argsort(np.asarray(local))[-10:]
        result.append(index)
    for j in result[0][0]:
        vocs.append(V[j])
    return vocs
    
random_error = []
    
    
    
    
        
    
    


   
    






















