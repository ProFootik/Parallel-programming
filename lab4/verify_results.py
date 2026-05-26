import numpy as np
import os

print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ CUDA")

sizes = [200, 400, 800, 1200, 1600, 2000]
block_configs = ["8x8", "16x16", "32x32"]

all_passed = True
verification_results = []

for size in sizes:
    print(f"\nПроверка размера {size}×{size}:")

    expected = np.loadtxt(f"data/matrices/expected_C_{size}.txt")
    
    for config in block_configs:
        result_file = f"results_cuda/result_{size}_{config}.txt"
        
        if os.path.exists(result_file):
            result = np.loadtxt(result_file)

            is_close = np.allclose(expected, result, rtol=1e-6, atol=1e-8)
            max_diff = np.max(np.abs(expected - result))
            
            if is_close:
                print(f"   Конфигурация {config}: пройдено (макс. разница: {max_diff:.2e})")
                verification_results.append((size, config, "PASSED", max_diff))
            else:
                print(f"   Конфигурация {config}: ошибка! Макс. разница: {max_diff:.2e}")
                verification_results.append((size, config, "FAILED", max_diff))
                all_passed = False
        else:
            print(f"   Файл {result_file} не найден")
            verification_results.append((size, config, "NOT_FOUND", 0))
            all_passed = False

with open("results_cuda/verification_results.txt", "w") as f:
    f.write("# Результаты верификации CUDA\n")
    f.write("# Размер\tКонфигурация\tСтатус\tМакс_ошибка\n")
    for size, config, status, diff in verification_results:
        f.write(f"{size}\t{config}\t{status}\t{diff:.6e}\n")

if all_passed:
    print(" ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
else:
    print(" ОБНАРУЖЕНЫ ОШИБКИ!")