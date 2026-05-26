import numpy as np
import sys

if len(sys.argv) != 4:
    print("Использование: python verify.py matrixA.txt matrixB.txt result.txt")
    sys.exit(1)

A = np.loadtxt(sys.argv[1])
B = np.loadtxt(sys.argv[2])
C_cpp = np.loadtxt(sys.argv[3])

C_py = A @ B

if np.allclose(C_cpp, C_py, atol=1e-5):
    print("Верификация успешна! Результаты совпадают.")
else:
    max_diff = np.max(np.abs(C_cpp - C_py))
    print("Верификация не пройдена!")
    print(f"Максимальное расхождение: {max_diff}")