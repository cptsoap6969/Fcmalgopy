import numpy as np
from sklearn.datasets import load_iris , load_wine, load_breast_cancer
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler


m = 2



def initialise_U(data, number_clusters):
	MAX = 10000.0
	U = []
	for i in range(0, len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0, number_clusters):
			dummy = random.randint(1,int(MAX))
			current.append(dummy)
			rand_sum += dummy
		for j in range(0, number_clusters):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U


def perfor(X , weight ,distance_matrix , c):
    temp_val = 0.0
    for i in range(0,c):
        for j in range(0, len(X)):
            temp_val += (weight[j][i] ** m) *  (distance_matrix[j][i]**2)
    
    return temp_val



def distance(point, center,alfa):
	if len(point) != len(center):
		return -1
	temp = 0.0
	for i in range(0, len(point)):
		temp += abs(point[i] - center[i]) ** 2
	temp = math.exp((-temp)/(2*alfa**2))
	return temp

def init_centers(weight,X,c):
    n = len(X)
    d = len(X[0])
    C = []
    for j in range(0, c):
        cluster_center = []
        for i in range(0, d):
            #top
            sum_num = 0.0
            #bottom
            sum_dum = 0.0
            for k in range(0, n):
                sum_num += (weight[k][j] ** m) * X[k][i]
                sum_dum += (weight[k][j] ** m)
            cluster_center.append(sum_num/sum_dum)
        C.append(cluster_center)
    
    return C

def init_matrix(X,C,alfa,c):
    n = len(X)
    distance_matrix =[]
    for i in range(0, n):
        temp_array = []
        for j in range(0, c):
            temp_array.append( 1 -distance(X[i], C[j],alfa))
        distance_matrix.append(temp_array)

    return distance_matrix

def update_weights(weight,distance_matrix,c,n):
    for j in range(0, c):	
        for i in range(0, n):
            temp_sum = 0.0
            for k in range(0, c):
                temp_sum += ((distance_matrix[i][j]) / (distance_matrix[i][k])) ** (1/(m-1))
            weight[i][j] = 1 / temp_sum
    
    return weight

def kfcm(X, weight , c,alfa):
    n = len(X)
    d = len(X[0])
    perf = []
    for it in range(50):
        C = init_centers(weight,X,c)
        distance_matrix = init_matrix(X,C,alfa,c)
        weight = update_weights(weight,distance_matrix,c,n)
        perf.append(perfor(X , weight ,distance_matrix , c))
    
    return weight, C, distance_matrix , perf


def graph_init(X , c,alfa=150):
    weight = initialise_U(X, c)
    weight, C, distance_matrix ,perf = kfcm(X , weight , c,alfa)
    distance_centers =[]
    for i in range(len(C)):
        temp_array = []
        for j in range(len(C)):
            temp_array.append(distance(C[i],C[j],alfa))
        distance_centers.append(temp_array)
    graphme =[]
    for i in range(0, len(weight)):
        maximum = max(weight[i])
        for j in range(0, len(weight[0])):
            if weight[i][j] != maximum:
                pass
            else:
                graphme.append(j)
                break
    return np.array(graphme) , c , X , weight, distance_matrix, m , perf , C,distance_centers
