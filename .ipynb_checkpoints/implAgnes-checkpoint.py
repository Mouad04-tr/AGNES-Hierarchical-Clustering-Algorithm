import numpy as np
import copy


def euclidean_distance(pointA, pointB):
    distance = np.linalg.norm(pointA - pointB)
    return distance

def create_proximity_matrix(data_points):
    num_points = len(data_points)
    proximity_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            distance = euclidean_distance(data_points[i], data_points[j])
            proximity_matrix[i][j] = distance
            proximity_matrix[j][i] = distance
    return proximity_matrix

def find_matrix_minimum(proximity_matrix):
    min_distance = np.inf
    min_row = None
    min_column = None
    num_points = len(proximity_matrix)
    for i in range(num_points):
        for j in range(i+1, num_points):
            if proximity_matrix[i][j] < min_distance:
                min_distance = proximity_matrix[i][j]
                min_row = i
                min_column = j
    return min_row, min_column

def update_proximity_matrix(proximity_matrix, merge_row, merge_column):
    new_proximity_matrix = copy.deepcopy(proximity_matrix)
    new_proximity_matrix[merge_row] = np.minimum(new_proximity_matrix[merge_row], new_proximity_matrix[merge_column])
    new_proximity_matrix[:, merge_row] = new_proximity_matrix[merge_row]
    new_proximity_matrix = np.delete(new_proximity_matrix, merge_column, axis=0)
    new_proximity_matrix = np.delete(new_proximity_matrix, merge_column, axis=1)
    return new_proximity_matrix

def update_cluster_ids(cluster_ids, merge_row, merge_column):
    cluster_ids[merge_row].extend(cluster_ids[merge_column])
    cluster_ids.pop(merge_column)
    return cluster_ids

def agnes(data_points, desired_number_of_clusters):
    cluster_ids = [[i] for i in range(len(data_points))]
    proximity_matrix = create_proximity_matrix(data_points)
    num_clusters = len(cluster_ids)
    
    while num_clusters > desired_number_of_clusters:
        print("Similarity Matrix:")
        print(proximity_matrix)
        
        merge_row, merge_column = find_matrix_minimum(proximity_matrix)
        proximity_matrix = update_proximity_matrix(proximity_matrix, merge_row, merge_column)
        cluster_ids = update_cluster_ids(cluster_ids, merge_row, merge_column)
        num_clusters = len(cluster_ids)
        
    datapoints_to_cluster_mapping = {}
    for which_cluster, current_cluster in enumerate(cluster_ids):
        for which_point in current_cluster:
            datapoints_to_cluster_mapping[which_point] = which_cluster
    
    return datapoints_to_cluster_mapping

# Données
data_points = np.array([
    [12, 13],
    [15, 17],
    [16, 20],
    [10, 11],
    [14, 16]
])

# Exécution de l'algorithme AGNES avec un nombre de clusters désiré
desired_number_of_clusters = 2
predicted_output_labels = agnes(data_points, desired_number_of_clusters)

# Affichage des résultats
for key, value in predicted_output_labels.items():
    print(f"Cluster ID for e{key + 1}: {value}")
