import numpy as np

def generate_matrix_file(filename, size, value_range=(-1000, 1000)):

    matrix = np.random.randint(value_range[0], value_range[1] + 1, (size, size))

    with open(filename, 'w') as file:
        file.write(str(size) + '\n')
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')


if __name__ == "__main__":

    n_matrix = 1000

    file_name_A = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\matrix_A.txt"
    file_name_B = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\matrix_B.txt"

    generate_matrix_file(file_name_A, n_matrix)
    generate_matrix_file(file_name_B, n_matrix)