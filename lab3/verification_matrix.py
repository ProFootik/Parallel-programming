import sys
import numpy as np

file_A = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\matrix_A.txt"
file_B = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\matrix_B.txt"
result_file = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\result_C.txt"


def read_matrix(filename):
    try:
        with open(filename, 'r') as f:
            N = int(f.readline().strip())
            data = [list(map(np.int64, f.readline().strip().split())) for _ in range(N)]
        return np.array(data, dtype=np.int64), N
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading matrix from file '{filename}': {e}")
        sys.exit(1)

def show_chek_result_matrix(filename, size, result_matrix):
    with open(filename, 'w') as file:
        file.write(str(size) + '\n')
        for row in result_matrix:
            file.write(' '.join(map(str, row)) + '\n')


def main():
    A_np, size_A = read_matrix(file_A)
    B_np, size_B = read_matrix(file_B)
    C_cpp, size_C = read_matrix(result_file)

    if size_A != size_B or size_A != size_C:
        print(f"Error: Matrix sizes mismatch (A:{size_A}, B:{size_B}, C:{size_C}).")
        sys.exit(1)

    C_expected = np.dot(A_np, B_np)

    if np.array_equal(C_cpp, C_expected):
        show_chek_result_matrix("C:\\Users\\Адель\\Desktop\\lab-1\\Data\\verification_result_C.txt", size_C, C_expected)
        print("C++ program results match NumPy results.")
    else:
        print("C++ program results don't match NumPy results")
        sys.exit(1)

if __name__ == "__main__":
    main()
