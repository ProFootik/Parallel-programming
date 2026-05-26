import numpy as np
import os

sizes = [200, 400, 800, 1200, 1600, 2000]

for n in sizes:
    a_file = f"matrixA_{n}.txt"
    b_file = f"matrixB_{n}.txt"
    if not os.path.exists(a_file):
        mat = np.random.uniform(-5, 5, (n, n))
        np.savetxt(a_file, mat, fmt='%.6f')
        np.savetxt(b_file, mat, fmt='%.6f')
        print(f"Сгенерированы матрицы {n}x{n}")