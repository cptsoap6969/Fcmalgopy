import numpy as np
import fcm


m=2



def PC(matrix):
    matrix_sum = 0
    total_cells = 0

    for row in matrix:
        for value in row:
            squared_value = value ** 2
            matrix_sum += squared_value
            total_cells += 1

    matrix_average = matrix_sum / total_cells

    return round(matrix_average,2)

def perfor(X , weight ,distance_matrix , c):
    temp_val = 0.0
    for i in range(0,c):
        for j in range(0, len(X)):
            temp_val += (weight[i][j] ** m) *  (distance_matrix[j][i]**2)
    
    return temp_val


def SC(U, c, distance_matrix, m,distance_centers):
        n = len(U)  # Number of data points
        U = np.array(U)
        """for i in range(len(distance_centers)):
            for j in range(len(distance_centers[0])):
                distance_centers[i][j] = distance_centers[i][j]**2"""

        distance_matrix = np.array(distance_matrix)
        graphme =[]
        for i in range(0, len(U)):
            maximum = max(U[i])
            for j in range(0, len(U[0])):
                if U[i][j] != maximum:
                    pass
                else:
                    graphme.append(j)

        sc = 0
        for j in range(c):
            sc += sc
            top = 0
            for i in range(n):
                top += (U[i, j] ** m) * distance_matrix[i, j]
            sc = top / (( 1 if graphme.count(j) == 0 else graphme.count(j) ) * sum(distance_centers[j]))

        # for j in range(c):
        #     for k in range(c):
        #         if distance_matrix[j, k] == 0.0:
        #             distance_matrix[j, k] = 1.0
        #         partition_coefficient /= (len(N[j])) * distance_matrix[j, k]

            

        return round(sc,2)

def group_array_values(array):
    grouped_values = {}
    for index, value in enumerate(array):
        if value not in grouped_values:
            grouped_values[value] = []
        grouped_values[value].append(index)
    similarity_matrix = list(grouped_values.values())
    return similarity_matrix

def calculate_similarity_percentage(matrix1, matrix2 , X):
    matching_elements = 0
    ordered = []
    for i in range(0, len(matrix1)):
        tmp = 0
        for j in range(0, len(matrix2)):
            if len(set(matrix1[i]) & set(matrix2[j])) > len(set(matrix1[i]) & set(matrix2[tmp])):
                if j not in ordered:
                    tmp = j
        ordered.append(tmp)

    n = len(matrix1)
    if len(matrix1) > len(matrix2):
        n = len(matrix2) 

    for i in range(n):
        matching_elements += len(set(matrix1[i]) & set(matrix2[ordered[i]]))

    similarity_percentage = (matching_elements / len(X)) * 100
    return round(similarity_percentage,2)