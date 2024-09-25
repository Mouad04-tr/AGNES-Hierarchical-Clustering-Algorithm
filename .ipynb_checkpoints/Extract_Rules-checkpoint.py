from math import sqrt
import copy


def euclidean_distance(pointA, pointB):
    distance = 0.0
    for index in range(len(pointA)):
        distance += (pointA[index] - pointB[index])**2
    return sqrt(distance)


def create_proximity_matrix(data_points):
    proximity_matrix = []
    for i in range(len(data_points)):
        row_matrix = [float("inf")] * len(data_points)
        for j in range(i):
            distance = euclidean_distance(data_points[i], data_points[j])
            row_matrix[j] = distance
        proximity_matrix.append(row_matrix)
    return proximity_matrix


def find_matrix_minimum(proximity_matrix):
    min_distance = float("inf")
    min_row = None
    min_column = None
    for i in range(len(proximity_matrix)):
        for j in range(i):
            if min_distance > proximity_matrix[i][j]:
                min_distance = proximity_matrix[i][j]
                min_row = i
                min_column = j
    return min_row, min_column


def update_proximity_matrix(proximity_matrix, merge_row, merge_column):
    new_proximity_matrix = copy.deepcopy(proximity_matrix)

    # Remove the row corresponding to merge_column
    del new_proximity_matrix[merge_column]

    # Update the remaining rows (except for merge_row) with the minimum distance
    for i, row in enumerate(new_proximity_matrix):
        if i != merge_row:
            # Adjust merge_row index if needed
            if i > merge_row:
                row[merge_row] = min(row[merge_row], row[merge_row + 1])
            # Remove the column corresponding to merge_row
            del row[merge_row]

    return new_proximity_matrix



def update_cluster_ids(cluster_ids, row, column):
    cluster_A = cluster_ids[row]
    cluster_B = cluster_ids[column]
    combined_cluster = cluster_A + cluster_B
    cluster_ids.pop(row)
    cluster_ids.pop(column)
    cluster_ids.insert(row, combined_cluster)
    return cluster_ids


def agnes(data_points, desired_number_of_clusters):
    cluster_ids = [[i] for i in range(len(data_points))]  # Initial clusters: each data point is in its own cluster
    proximity_matrix = create_proximity_matrix(data_points)
    number_of_clusters = len(cluster_ids)

    while number_of_clusters > desired_number_of_clusters:
        row, column = find_matrix_minimum(proximity_matrix)
        proximity_matrix = update_proximity_matrix(proximity_matrix, row, column)
        cluster_ids = update_cluster_ids(cluster_ids, row, column)
        number_of_clusters = len(cluster_ids)

    datapoints_to_cluster_mapping = {}
    for which_cluster, current_cluster in enumerate(cluster_ids):
        for which_point in current_cluster:
            datapoints_to_cluster_mapping[which_point] = current_cluster[0]

    predicted_labels = []
    for key in sorted(datapoints_to_cluster_mapping.keys()):
        predicted_labels.append(datapoints_to_cluster_mapping[key])

    return predicted_labels


def read_dataset(filename):
    data_points = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split()
            data_points.append([float(parts[1]), float(parts[2])])  # Extracting note1 and note2 values
    return data_points


if __name__ == '__main__':
    data_points = read_dataset("data.txt")
    desired_number_of_clusters = 2  # Adjust as needed
    
    predicted_output_labels = agnes(data_points, desired_number_of_clusters)

    # You can evaluate the results if you have ground truth labels
    # true_labels = read_ground_truth_labels("ground_truth_labels.txt")
    # accuracy = check_accuracy(true_labels, predicted_output_labels)
    # print("Accuracy:", accuracy)
