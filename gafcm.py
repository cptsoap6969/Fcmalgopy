import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import math
import random


m = 2
MAX = 10000.0
ε = 0.01

def update_weights(X,c,weight , distance_matrix, m):
    condition = False
    n = len(X)
    d = len(X[0])
    for j in range(0, c):	
        for i in range(0, n):
            temp_sum = 0.0
            for k in range(0, c):
                if  1 - distance_matrix[i][k] == 0:
                    condition = True
                else :
                    temp_sum += (1 - distance_matrix[i][k]) ** (-1/(m-1))
            if condition :
                for j2 in range(0,c):
                    weight[i][j2] = 0
                weight[i][j] = 1
                condition = False
            else :
                weight[i][j] = ((1 - distance_matrix[i][j])**(-1/(m-1))) / temp_sum
    return weight

def update_centers(weight,X ,distance_matrix , c):
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
                sum_num += (weight[k][j]) * X[k][i] * distance_matrix[k][j]
                sum_dum += (weight[k][j] * distance_matrix[k][j])
            cluster_center.append(sum_num/sum_dum)
        C.append(cluster_center)
    return C

def calc_matrix(X , C ,c):
    distance_matrix =[]
    n = len(X)
    for i in range(0, n):
        temp_array = []
        for j in range(0, c):
            temp_array.append(distance(X[i], C[j]))
        distance_matrix.append(temp_array)
    
    return distance_matrix

def fcm_obj(X,C ,c ,m):
    weight, distance_matrix, perf = fcm(X,C ,c ,m)
    # distance_matrix = calc_matrix(X ,C ,c)
    # weight = update_weights(X,c,weight ,distance_matrix ,m)

    temp_val = 0.0
    for i in range(0,c):
        for j in range(0, len(X)):
            temp_val += (weight[j][i] ** m) *  (distance_matrix[j][i]**2)

    
    return [temp_val, weight, distance_matrix, perf]




def fitness_function(X,C ,c, m):
    objective = fcm_obj(X ,C,c ,m)
    return objective

def crossover(parent1, parent2):
    child = np.empty(parent1.shape)
    rows, cols = parent1.shape
    
    for row in range(rows):
        crossover_point = np.random.randint(1, cols)
        child[row, :crossover_point] = parent1[row, :crossover_point]
        child[row, crossover_point:] = parent2[row, crossover_point:]
    
    return child

def mutation(array, mutation_rate):
    mutated_array = np.copy(array)
    shape = array.shape
    
    for _ in range(int(mutation_rate * shape[0] * shape[1])):
        row_index = np.random.randint(shape[0])
        col_index = np.random.randint(shape[1])
        
        if np.random.rand() < 0.5:
            mutated_array[row_index, col_index] += np.random.uniform(0, 0.1)
        else:
            mutated_array[row_index, col_index] -= np.random.uniform(0, 0.1)
    
    return mutated_array



def ga_fcm(X,c,n_population , n_generations):
  prob_crossover = 0.7
  prob_mutation = 1.0
  population = [X[np.random.choice(len(X), size=c, replace=False)] for _ in range(n_population*2)]

  centers = []
  fitness_old = MAX
  perf = []
  bestPerformance = []
  for i in range(n_generations):
      fitt = [fitness_function(X,centers ,c, m) for centers in population]
      fitness_values = [i[0] for i in fitt]
      population = [x for _, x in sorted(zip(fitness_values, population))]
      weight =[i[1] for i in fitt]
      disMatrix =[i[2] for i in fitt]
      Perf =[i[3] for i in fitt]
      if i == 0 :
        fitness_old = fitness_values[np.argmin(fitness_values)]
        weights = weight[np.argmin(fitness_values)]
        distance_matrix = disMatrix[np.argmin(fitness_values)]
        centers = population[np.argmin(fitness_values)]
        bestPerformance = Perf[np.argmin(fitness_values)]
      if fitness_old > fitness_values[np.argmin(fitness_values)]:
          fitness_old = fitness_values[np.argmin(fitness_values)]
          weights = weight[np.argmin(fitness_values)]
          distance_matrix = disMatrix[np.argmin(fitness_values)]
          centers = population[np.argmin(fitness_values)]
          bestPerformance = Perf[np.argmin(fitness_values)]
      perf.append(fitness_values[np.argmin(fitness_values)])

      num_parents = len(population) // 2
      parents = population[:num_parents]

      """for j in range(n_population):
          k1 = np.random.randint(0, n_population)
          k2 = np.random.randint(0, n_population)
          if fitness_values[k1] > fitness_values[k2]:
              parents.append(population[k1])
          else:
              parents.append(population[k2])"""
      offspring = []
      for j in range(n_population // 2):
          parent1 = parents[j]
          parent2 = parents[n_population  - j - 1]
          
          if np.random.uniform() < prob_crossover:
              child1 = crossover(parent1, parent2)
              child2 = crossover(parent2, parent1)
          else:
              child1 = parent1
              child2 = parent2
          
          offspring.append(mutation(child1, prob_mutation))
          offspring.append(mutation(child2, prob_mutation))

      num_remaining = n_population * 2 - len(offspring)
      
      population =  offspring + population[:num_remaining]

  if len(centers) == 0:
      centers = population[np.argmin(fitness_values)]
  return centers , weights, distance_matrix, perf

def fcm_run(X,C ,c ,m):
    weight = np.zeros(shape=X.shape)
    distance_matrix = calc_matrix(X ,C,c)
    weight = update_weights(X,c,weight ,distance_matrix ,m)
    
    return weight, distance_matrix

"""def fcm(X, C , c, m):
    n = len(X)
    d = len(X[0])
    perf = []
    weight  = np.zeros((n, d))
    while True: # Total number of iterations
        distance_matrix =[]
        for i in range(0, n):
            temp_array = []
            for j in range(0, c):
                x =  1 if distance(X[i], C[j]) == 0 else distance(X[i], C[j])
                temp_array.append( x )
            distance_matrix.append(temp_array)

        for j in range(0, c):	
            for i in range(0, n):
                temp_sum = 0.0
                for k in range(0, c):
                    temp_sum += ((distance_matrix[i][j]) / (distance_matrix[i][k])) ** (1/(m-1))
                weight[i][j] = 1 / temp_sum
        C_new = []
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
            C_new.append(cluster_center)
            
        if check_stopping_criterion(C, C_new ,ε):
            break
        C = C_new
        
        perf.append(perfor(X , weight ,distance_matrix , c))
    
    return weight, distance_matrix , perf"""

def fcm(X, C , c, m):
    n = len(X)
    d = len(X[0])
    perf = []
    weight  = np.zeros((n, d))
    for i in range(50): # Total number of iterations
        distance_matrix =[]
        for i in range(0, n):
            temp_array = []
            for j in range(0, c):
                x =  1 if distance(X[i], C[j]) == 0 else distance(X[i], C[j])
                temp_array.append( x )
            distance_matrix.append(temp_array)

        for j in range(0, c):	
            for i in range(0, n):
                temp_sum = 0.0
                for k in range(0, c):
                    temp_sum += ((distance_matrix[i][j]) / (distance_matrix[i][k])) ** (1/(m-1))
                weight[i][j] = 1 / temp_sum
        C_new = []
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
            C_new.append(cluster_center)
            
        """if check_stopping_criterion(C, C_new ,ε):
            break"""
        C = C_new
        
        perf.append(perfor(X , weight ,distance_matrix , c))
    
    return weight, distance_matrix , perf



def graph_init(X,c,n_population , n_generations):
    C , weight, distance_matrix, perf = ga_fcm(X,c,n_population , n_generations)
    # weight, distance_matrix, perf = fcm(X,C ,c ,m)
    distance_centers =[]
    for i in range(len(C)):
        temp_array = []
        for j in range(len(C)):
            temp_array.append(distance(C[i],C[j]))
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
    return np.array(graphme) , c , X, weight, distance_matrix, m, perf , C,distance_centers

def perfor(X , weight ,distance_matrix , c):
    temp_val = 0.0
    for i in range(0,c):
        for j in range(0, len(X)):
            temp_val += (weight[j][i] ** m) *  (distance_matrix[j][i]**2)
    
    return temp_val


def distance(point, center):
    if len(point) != len(center):
        return -1
    temp_sum = 0.0
    for i in range(0, len(point)):
        temp_sum += abs(point[i] - center[i]) ** 2
    return math.sqrt(temp_sum)


def check_stopping_criterion(V_old, V_new, ε):
    c = len(V_old)
    d = 0
    for i in range(c):
        d += distance(V_old[i], V_new[i])
    return d / c < ε