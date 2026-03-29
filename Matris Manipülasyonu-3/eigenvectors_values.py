import numpy as np
import pandas as pd


# Referans alınan yaklaşım:
# https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
# Bu kod, o fikri daha okunabilir ve karşılaştırma yapılabilir şekilde yeniden düzenler.


def get_dimensions(matrix):
    return [len(matrix), len(matrix[0])]


def list_multiply(list1, list2):
    result = [0 for _ in range(len(list1) + len(list2) - 1)]
    for i in range(len(list1)):
        for j in range(len(list2)):
            result[i + j] += list1[i] * list2[j]
    return result


def list_add(list1, list2, sub=1):
    return [i + (sub * j) for i, j in zip(list1, list2)]


def identity_matrix(dimensions):
    matrix = [[0 for _ in range(dimensions[1])] for _ in range(dimensions[0])]
    for i in range(dimensions[0]):
        matrix[i][i] = 1
    return matrix


def characteristic_equation(matrix):
    dimensions = get_dimensions(matrix)
    return [[[a, -b] for a, b in zip(row, eye_row)]
            for row, eye_row in zip(matrix, identity_matrix(dimensions))]


def determinant_equation(matrix, excluded=None):
    if excluded is None:
        excluded = [1, 0]

    dimensions = get_dimensions(matrix)

    if dimensions == [2, 2]:
        tmp = list_add(
            list_multiply(matrix[0][0], matrix[1][1]),
            list_multiply(matrix[0][1], matrix[1][0]),
            sub=-1
        )
        return list_multiply(tmp, excluded)

    new_matrices = []
    excluded_values = []
    exclude_row = 0

    for exclude_column in range(dimensions[1]):
        tmp = []
        excluded_values.append(matrix[exclude_row][exclude_column])

        for row in range(1, dimensions[0]):
            tmp_row = []
            for column in range(dimensions[1]):
                if (row != exclude_row) and (column != exclude_column):
                    tmp_row.append(matrix[row][column])
            tmp.append(tmp_row)
        new_matrices.append(tmp)

    determinant_equations = [
        determinant_equation(new_matrices[j], excluded_values[j])
        for j in range(len(new_matrices))
    ]

    max_len = max(len(eq) for eq in determinant_equations)
    padded = [eq + [0] * (max_len - len(eq)) for eq in determinant_equations]
    dt_equation = [sum(items) for items in zip(*padded)]
    return dt_equation


def find_eigenvalues_without_eig(matrix):
    dt_equation = determinant_equation(characteristic_equation(matrix))
    roots = np.roots(dt_equation[::-1])
    roots = np.real_if_close(roots, tol=1000)
    return roots, dt_equation


def nullspace_vector(B):
    _, _, vt = np.linalg.svd(B)
    v = vt[-1, :]
    v = np.real_if_close(v, tol=1000)
    idx = np.argmax(np.abs(v))
    if v[idx] < 0:
        v = -v
    norm = np.linalg.norm(v)
    return v / norm


def find_eigenvectors_from_eigenvalues(matrix, eigenvalues):
    A = np.array(matrix, dtype=float)
    vectors = []
    for lam in eigenvalues:
        vectors.append(nullspace_vector(A - lam * np.eye(A.shape[0])))
    return np.column_stack(vectors)


def compare_results(matrix):
    A = np.array(matrix, dtype=float)

    manual_values, coeffs = find_eigenvalues_without_eig(matrix)
    manual_values = np.array(sorted(np.real_if_close(manual_values), reverse=True), dtype=float)
    manual_vectors = find_eigenvectors_from_eigenvalues(matrix, manual_values)

    numpy_values, numpy_vectors = np.linalg.eig(A)
    numpy_values = np.real_if_close(numpy_values, tol=1000)
    order = np.argsort(-numpy_values)
    numpy_values = numpy_values[order]
    numpy_vectors = np.real_if_close(numpy_vectors[:, order], tol=1000)

    comparison = pd.DataFrame({
        "Elle/Referans Yöntemi Özdeğerleri": manual_values,
        "NumPy eig Özdeğerleri": numpy_values,
        "Mutlak Fark": np.abs(manual_values - numpy_values)
    })

    return coeffs, manual_values, manual_vectors, numpy_values, numpy_vectors, comparison


if __name__ == "__main__":
    A = [[6, 1, -1],
         [0, 7, 0],
         [3, -1, 2]]

    coeffs, manual_vals, manual_vecs, np_vals, np_vecs, table = compare_results(A)

    print("Karakteristik polinom katsayıları [sabit, λ, λ^2, λ^3]:")
    print(coeffs)
    print("\nReferans yaklaşımla bulunan özdeğerler:")
    print(manual_vals)
    print("\nNumPy eig ile bulunan özdeğerler:")
    print(np_vals)
    print("\nKarşılaştırma tablosu:")
    print(table.to_string(index=False))
