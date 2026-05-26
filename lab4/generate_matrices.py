import numpy as np
import os

os.makedirs("data/matrices", exist_ok=True)

sizes = [200, 400, 800, 1200, 1600, 2000]

print("Генерация матриц (CUDA)")
print("=" * 60)

for size in sizes:
    print(f"Генерация матриц размера {size}×{size}...")

    A = np.random.randint(-10, 10, (size, size)).astype(np.float64)
    B = np.random.randint(-10, 10, (size, size)).astype(np.float64)

    np.savetxt(f"data/matrices/matrix_A_{size}.txt", A, fmt="%.6f")
    np.savetxt(f"data/matrices/matrix_B_{size}.txt", B, fmt="%.6f")

    C_expected = A @ B
    np.savetxt(f"data/matrices/expected_C_{size}.txt", C_expected, fmt="%.6f")

print(" Генерация завершена! Матрицы сохранены в папку 'data/matrices/'")