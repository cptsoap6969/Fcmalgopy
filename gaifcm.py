import numpy as np
import random
import math
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import func




ε=0.0000001
MAX = 10000.0


def convert_to_ifs(x, min_value, max_value, lamda):
    a = (x - min_value)/(max_value - min_value)
    b = (1 - a**lamda)**(1/lamda)
    return {'membership': a, 'non_membership': b, 'hesitation': 1-a-b}

def convert_dataset_to_ifs(X, lamda):
    n, p = X.shape
    X_ifs = np.zeros((n, p), dtype=np.dtype('O'))
    
    for i in range(p):
        max_value = np.max(X[:, i])
        min_value = np.min(X[:, i])
        for j in range(n):
            X_ifs[j, i] = convert_to_ifs(X[j, i], min_value, max_value ,lamda)
    
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

def calculate_membership_degrees(V, X, m):
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
    return np.transpose(U), D

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

def perfor(X , weight ,distance_matrix , c, m):
    temp_val = 0.0
    for i in range(0,c):
        for j in range(0, len(X)):
            temp_val += (weight[i][j] ** m) *  (distance_matrix[j][i]**2)
    
    return temp_val

def ga_ifcm(X_ifs, c, ε, m):
    V = initialize_seeds(c, X_ifs)
    k = 0
    perf = []
    g = 0
    for i in range(50):
        U, distance_matrix = calculate_membership_degrees(V, X_ifs,m)
        V_new = update_cluster_centers(U, X_ifs)
        """if check_stopping_criterion(V, V_new ,ε):
            g = check_stopping_criterio_obj(V, V_new,ε)
            break"""
        perf.append(perfor(X_ifs , U ,distance_matrix , c, m))
        V = V_new
        k += 1
    
    return V, np.transpose(U) , g, distance_matrix, perf


# def ifcm(X_ifs, c, ε, m,V):
#     k = 0
#     perf = []
#     while True :
#         U, D = calculate_membership_degrees(V, X_ifs,m)
#         V_new = update_cluster_centers(U, X_ifs)
#         if check_stopping_criterion(V, V_new ,ε):
#             break
        
#         perf.append(perfor(X_ifs , U ,D , c, m))
#         V = V_new
#         k += 1
    # return V, np.transpose(U), D , perf

def ifcm(X_ifs, c, ε, m, V):
    k = 0
    perf = []
    for i in range(50):
        U, D = calculate_membership_degrees(V, X_ifs)
        V_new = update_cluster_centers(U, X_ifs, f)
        """if check_stopping_criterion(V, V_new ,ε):
            break"""
        
        perf.append(perfor(X_ifs , U ,D , c))
        V = V_new
        k += 1
    return V, np.transpose(U), D , perf

def check_stopping_criterio_obj(V_old, V_new, ε):
    c = V_old.shape[0]
    d = 0
    for i in range(c):
        d += d1(V_old[i], V_new[i])
    return d / c



def fitness_function(X,F,c, lamda, m):
    
    
    X_ifs = convert_dataset_to_ifs(X, lamda)
    # Calculate the objective function for the fuzzy C-means algorithm using the given cluster centers

    V , weight ,objective,distance_matrix, perf = ga_ifcm(X_ifs, c, ε, m)

    graphme =[]
    for i in range(0, len(weight)):
        maximum = max(weight[i])
        for j in range(0, len(weight[0])):
            if weight[i][j] != maximum:
                pass
            else:
                graphme.append(j)
    grouped_results = func.group_array_values(F)
    grouped_values = func.group_array_values(graphme)
    similarity_percentage = func.calculate_similarity_percentage(grouped_results, grouped_values,X)
    
    # Return the negative of the objective function as the fitness score

    return [similarity_percentage , V, weight, distance_matrix, perf]

def crossover(parent1, parent2):
    # Get the length of the parents and the crossover point
    #parent shape = [X , X]
    n = len(parent1)
    k = np.random.randint(1, n)
    
    # Create the child by combining the parents at the crossover point
    child = np.zeros(shape=len(parent1))
    chance = random.random()
    if random.random() > 0.5:
      child[0] = parent1[0]
      child[1] = parent2[1]
    else:
      child[0] = parent2[0]
      child[1] = parent1[1]
    
    return child

def mutation(individual, prob_mut):
    # Create a copy of the individual
    mutated_individual = individual.copy()
    
    # Loop through each gene and mutate with the given probability
    for i in range(len(mutated_individual)):
        if np.random.uniform() < prob_mut:
            mutated_individual[i] += np.random.normal(scale=0.1)
    
    return mutated_individual



def gaifcm(X,F,c,m_start,m_end,lamda_start,lamda_end ,n_population , n_generations):
    prob_crossover = 0.8
    prob_mutation = 0.2
    # Define the initial population for the genetic algorithm
    lamda_gen  =0
    m_gen = 0
    fitness_old = 0
    Centers  = []
    perf = []

    population = [[random.uniform(lamda_start, lamda_end) , random.uniform(m_start,m_end)] for _ in range(n_population*2)]
    for i in range(n_generations):
        #weights[0] = lamda
        #weights[1] = m
        fitt = [fitness_function(X,F,c ,weights[0], weights[1] ) for weights in population]
        fitness_values = [i[0] for i in fitt]
        population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
        cen = [i[1] for i in fitt]
        weight =[i[2] for i in fitt]
        disMatrix =[i[3] for i in fitt]
        performance =[i[4] for i in fitt]

        if i == 0 :
            fitness_old = fitness_values[np.argmax(fitness_values)]
            lamda_gen = population[np.argmax(fitness_values)][0]
            m_gen = population[np.argmax(fitness_values)][1]
            Centers = cen[np.argmax(fitness_values)]
            Weights = weight[np.argmax(fitness_values)]
            DistanceMatrix = disMatrix[np.argmax(fitness_values)]
            Performance = performance[np.argmax(fitness_values)]

        if fitness_old < fitness_values[np.argmax(fitness_values)]:
            fitness_old = fitness_values[np.argmax(fitness_values)]
            lamda_gen = population[np.argmax(fitness_values)][0]
            m_gen = population[np.argmax(fitness_values)][1]
            Centers = cen[np.argmax(fitness_values)]
            Weights = weight[np.argmax(fitness_values)]
            DistanceMatrix = disMatrix[np.argmax(fitness_values)]
            Performance = performance[np.argmax(fitness_values)]
        perf.append(fitness_values[np.argmax(fitness_values)])
        
        # Select the parents for the next generation using tournament selection
        num_parents = len(population) // 2
        parents = population[:num_parents]
        """for j in range(n_population):
            k1 = np.random.randint(0, n_population)
            k2 = np.random.randint(0, n_population)
            if fitness_values[k1] > fitness_values[k2]:
                parents.append(population[k1])
            else:
                parents.append(population[k2])"""
        
        # Create the next generation by applying crossover and mutation to the parents
        offspring = []
        for j in range(n_population // 2):
            parent1 = parents[j]
            parent2 = parents[n_population - j - 1]
            
            if np.random.uniform() < prob_crossover:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
            else:
                child1 = parent1
                child2 = parent2
            
            offspring.append(mutation(child1, prob_mutation))
            offspring.append(mutation(child2, prob_mutation))
        
        # Replace the old population with the new offspring
        num_remaining = n_population * 2 - len(offspring)
      
        population =  offspring + population[:num_remaining]


    return lamda_gen , m_gen , Centers, Weights, DistanceMatrix, perf
 




def graph_init(X,c,F,m_start,m_end,lamda_start,lamda_end,n_population , n_generations):
    lamda , m , V_final, weight, distance_matrix, perf = gaifcm(X,F,c,m_start,m_end,lamda_start,lamda_end,n_population , n_generations)

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
    return np.array(graphme) , c , X, weight, distance_matrix, m,lamda , perf , V_final,distance_centers
