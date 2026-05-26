import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]
nps = [1, 2, 4, 8]
results = []

for n in sizes:
    a_file = f"matrixA_{n}.txt"
    b_file = f"matrixB_{n}.txt"
    res_file = f"result_{n}.txt"

    if not os.path.exists(a_file):
        import numpy as np
        mat = np.random.uniform(-5, 5, (n, n))
        np.savetxt(a_file, mat, fmt='%.6f')
        np.savetxt(b_file, mat, fmt='%.6f')

    for np_val in nps:
        try:
            proc = subprocess.run(["mpirun", "-np", str(np_val), "--oversubscribe", "./matrix_mul",
                                           a_file, b_file, res_file],
                                           capture_output=True, text=True, timeout=600)

            output = proc.stdout + proc.stderr

            time_match = re.search(r"Время выполнения: ([\d.]+) секунд", output)
            if time_match:
                t = float(time_match.group(1))
                vol = n * n * n
                results.append({"N": n, "NP": np_val, "Time_s": t, "Volume": vol})
                print(f"N={n:4d} | NP={np_val:2d} | Время: {t:8.4f} с")
            else:
                print(f"N={n} NP={np_val} — не удалось прочитать время")
                if output.strip():
                    print("Вывод программы:")
                    print(output.strip()[:800])

        except Exception as e:
            print(f"N={n} NP={np_val} — ошибка: {e}")

df = pd.DataFrame(results)
df.to_csv("results_mpi.csv", index=False)
print("\nРезультаты сохранены в results_mpi.csv")

# График
plt.figure(figsize=(12, 7))
for np_val in sorted(df["NP"].unique()):
    sub = df[df["NP"] == np_val]
    plt.plot(sub["N"], sub["Time_s"], marker='o', linewidth=2, label=f"{np_val} процессов")

plt.xlabel("Размер матрицы N")
plt.ylabel("Время выполнения (секунды)")
plt.title("Умножение матриц MPI (наивный алгоритм)")
plt.grid(True)
plt.legend()
plt.savefig("time_vs_n_mpi.png")
plt.show()
print("График сохранён: time_vs_n_mpi.png")