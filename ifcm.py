import numpy as np
from sklearn.datasets import load_iris , load_wine
import matplotlib.pyplot as plt

def convert_to_ifs(x, min_value, max_value):
    lamda = 0.5
    a = (x - min_value)/(max_value - min_value)
    b = (1 - a**lamda)**(1/lamda)
    return {'membership': a, 'non_membership': b, 'hesitation': 1-a-b}

def convert_dataset_to_ifs(X):
    n, p = X.shape
    X_ifs = np.zeros((n, p), dtype=np.dtype('O'))
    
    for i in range(p):
        max_value = np.max(X[:, i])
        min_value = np.min(X[:, i])
        for j in range(n):
            X_ifs[j, i] = convert_to_ifs(X[j, i], min_value, max_value)
    
    return X_ifs

def d1(A, B):
    n = len(A)
    d = 0
    for i in range(n):
        d += ((A[i]['membership'] - B[i]['membership']) ** 2 +
              (A[i]['non_membership'] - B[i]['non_membership']) ** 2 +
              (A[i]['hesitation'] - B[i]['hesitation']) ** 2)
    return (d / (2*n)) ** 0.5





def f(Z, w):
    n = Z.shape[0]
    p = Z.shape[1]
    V = np.zeros(p, dtype=np.dtype('O'))
    
    for j in range(p):
        V[j] = {'membership': 0, 'non_membership': 0, 'hesitation': 0}
        for i in range(n):
            V[j]['membership'] += w[i] * Z[i, j]['membership']
            V[j]['non_membership'] += w[i] * Z[i, j]['non_membership']
            V[j]['hesitation'] += w[i] * Z[i, j]['hesitation']
    return V

def initialize_seeds(c, X):
    n = X.shape[0]
    indices = np.random.choice(n, size=c, replace=False)
    V = X[indices]
    return V

def calculate_membership_degrees(V, X, m=2):
    c = V.shape[0]
    n = X.shape[0]
    U = np.zeros((n, c))
    D = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            D[i, j] = d1(X[i], V[j])
    for i in range(n):
        for j in range(c):
            array = D[i]
            index = np.where(array == 0)
            if len(array[index]):
              value=index[0]
              U[i, value[0] ] = 1
              break
            else:
              x=0
              for k in range(c):
                x += (D[i, j]/D[i, k])**(2/(m-1))
              U[i, j] = 1/ (x)
    return np.transpose(U),D

def update_cluster_centers(U, X):
    c = U.shape[0]
    n = X.shape[0]
    V = np.zeros((c, X.shape[1]), dtype=np.dtype('O'))
    w = np.zeros( X.shape[0], dtype=np.dtype('O'))
    for i in range(c):
      for j in range (n):
        w[j]= U[i, j]/ np.sum(U[i])
      V[i] = f(X, w)
    return V

def check_stopping_criterion(V_old, V_new, ε):
    c = V_old.shape[0]
    d = 0
    for i in range(c):
        d += d1(V_old[i], V_new[i])
    return d / c < ε

def check_perf(V_old, V_new):
    c = V_old.shape[0]
    d = 0
    for i in range(c):
        d += d1(V_old[i], V_new[i])
    return d / c

def perfor(X , weight ,distance_matrix , c):
    temp_val = 0.0
    for i in range(0,c):
        for j in range(0, len(X)):
            temp_val += (weight[i][j] ** m) *  (distance_matrix[j][i]**2)
    
    return temp_val


def ifcm(X_ifs, c, ε, m):
    V = initialize_seeds(c, X_ifs)
    k = 0
    perf = []
    for i in range(50):
        U, D = calculate_membership_degrees(V, X_ifs)
        V_new = update_cluster_centers(U, X_ifs)
        """if check_stopping_criterion(V, V_new ,ε):
            break"""
        
        perf.append(perfor(X_ifs , U ,D , c))
        V = V_new
        k += 1
    return V, np.transpose(U), D , perf

ε=0.000000001
m=2

def graph_init(X , c):
    X_ifs = convert_dataset_to_ifs(X)
    V_final, weight, D , perf=ifcm(X_ifs,c , ε, m)
    distance_centers =[]
    for i in range(len(V_final)):
        temp_array = []
        for j in range(len(V_final)):
            temp_array.append(d1(V_final[i],V_final[j]))
        distance_centers.append(temp_array)
    graphme =[]
    for i in range(0, len(weight)):
        maximum = max(weight[i])
        for j in range(0, len(weight[0])):
            if weight[i][j] != maximum:
                pass
            else:
                graphme.append(j)
    return np.array(graphme) , c , X, weight, D, m, perf , V_final,distance_centers



