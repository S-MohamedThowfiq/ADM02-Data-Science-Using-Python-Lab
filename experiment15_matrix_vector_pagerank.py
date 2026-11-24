"""
EXPERIMENT 15 - Matrix-Vector Multiplication using MapReduce
Computes x = Mv for a sparse 5-page link matrix, checks the result with
NumPy, and repeats the multiplication to illustrate PageRank iteration.

Requires: pip install numpy
Run: python experiment15_matrix_vector_pagerank.py
"""

import numpy as np
from collections import defaultdict


def map_matrix_entries(entries, vector):
    """Emit (row_index, matrix_value * vector[column_index])."""
    output = []
    for row, column, value in entries:
        output.append((row, value * vector[column]))
    return output


def shuffle_and_sort(mapped):
    grouped = defaultdict(list)
    for row, value in mapped:
        grouped[row].append(value)
    return sorted(grouped.items())


def reduce_rows(groups, size):
    result = np.zeros(size, dtype=float)
    for row, values in groups:
        result[row] = sum(values)
    return result


def main():
    matrix = np.array([
        [0,     0.5, 0,   0,   1/3],
        [1/3,   0,   0,   0.5, 0],
        [1/3,   0,   0,   0.5, 1/3],
        [1/3,   0,   1.0, 0,   1/3],
        [0,     0.5, 0,   0,   0],
    ], dtype=float)
    vector = np.full(5, 0.2)

    entries = [
        (i, j, matrix[i, j])
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if matrix[i, j] != 0
    ]

    # Store sparse matrix and vector in text files, as separate HDFS inputs.
    with open("matrix.txt", "w", encoding="utf-8") as file:
        for i, j, value in entries:
            file.write(f"{i}\t{j}\t{value}\n")
    with open("vector.txt", "w", encoding="utf-8") as file:
        for j, value in enumerate(vector):
            file.write(f"{j}\t{value}\n")

    print("MATRIX-VECTOR MULTIPLICATION USING MAPREDUCE")
    print("Matrix M:\n", np.round(matrix, 3))
    print("Vector v:", vector)
    print(f"Non-zero matrix entries: {len(entries)} of {matrix.size}")

    print("\n--- MAP PHASE ---")
    mapped = map_matrix_entries(entries, vector)
    for (i, j, value), (key, result) in zip(entries, mapped):
        print(f"({i}, {j}, {value:.3f}) -> ({key}, {result:.4f})")

    print("\n--- SHUFFLE AND SORT PHASE ---")
    groups = shuffle_and_sort(mapped)
    for row, values in groups:
        print(f"Row {row}: {[round(value, 4) for value in values]}")

    print("\n--- REDUCE PHASE ---")
    result = reduce_rows(groups, len(vector))
    for i, value in enumerate(result):
        print(f"x[{i}] = {value:.4f}")

    numpy_result = matrix @ vector
    print("\nMapReduce result:", np.round(result, 4))
    print("NumPy M @ v:", np.round(numpy_result, 4))
    print("Results match:", np.allclose(result, numpy_result))

    print("\n--- PAGERANK POWER ITERATION ---")
    rank = vector.copy()
    for iteration in range(1, 31):
        rank = matrix @ rank
        if iteration in (1, 2, 5, 30):
            print(f"Iteration {iteration:2}: {np.round(rank, 4)}")
    print("Most important page: P" + str(int(np.argmax(rank))))


if __name__ == "__main__":
    main()
