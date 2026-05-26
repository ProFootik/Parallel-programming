import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

sizes = []
configs = []
times = []
gflops = []

with open("results_cuda/performance_results.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            sizes.append(int(parts[0]))
            configs.append(parts[1])
            times.append(float(parts[2]))
            gflops.append(float(parts[3]))

unique_sizes = sorted(set(sizes))
unique_configs = sorted(set(configs))
colors = {'8x8': '#1f77b4', '16x16': '#ff7f0e', '32x32': '#2ca02c'}

print("Построение графиков...")

# ============================================
# График 1: Зависимость времени от размера матрицы
# ============================================
plt.figure(figsize=(10, 6))

for config in unique_configs:
    config_times = [times[i] for i in range(len(sizes)) if configs[i] == config]
    plt.plot(unique_sizes, config_times, 'o-', 
             linewidth=2.5, markersize=8, 
             label=f'Конфигурация {config}', 
             color=colors[config])

plt.xlabel('Размер матрицы (n)', fontsize=14)
plt.ylabel('Время выполнения (секунды)', fontsize=14)
plt.title('Зависимость времени выполнения от размера матрицы\n(CUDA, Tesla T4)', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper left')
plt.xticks(unique_sizes, fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.savefig('results_cuda/fig1_time_vs_size.png', dpi=200, bbox_inches='tight')
plt.show()
print(" fig1_time_vs_size.png сохранён")

# ============================================
# График 2: Зависимость производительности от размера матрицы
# ============================================
plt.figure(figsize=(10, 6))

for config in unique_configs:
    config_gflops = [gflops[i] for i in range(len(sizes)) if configs[i] == config]
    plt.plot(unique_sizes, config_gflops, 's-', 
             linewidth=2.5, markersize=8, 
             label=f'Конфигурация {config}', 
             color=colors[config])

plt.xlabel('Размер матрицы (n)', fontsize=14)
plt.ylabel('Производительность (GFLOPS)', fontsize=14)
plt.title('Зависимость производительности от размера матрицы\n(CUDA, Tesla T4)', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right')
plt.xticks(unique_sizes, fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.savefig('results_cuda/fig2_gflops_vs_size.png', dpi=200, bbox_inches='tight')
plt.show()
print(" fig2_gflops_vs_size.png сохранён")

# ============================================
# График 3: Логарифмическая шкала времени
# ============================================
plt.figure(figsize=(10, 6))

for config in unique_configs:
    config_times = [times[i] for i in range(len(sizes)) if configs[i] == config]
    plt.loglog(unique_sizes, config_times, 'o-', 
               linewidth=2.5, markersize=8, 
               label=f'Конфигурация {config}', 
               color=colors[config])

n_cubed = [times[-1] * (s / unique_sizes[-1])**3 for s in unique_sizes]
plt.loglog(unique_sizes, n_cubed, 'k--', 
           linewidth=2, alpha=0.7, 
           label='Теоретическая O(n³)')

plt.xlabel('Размер матрицы n (логарифмическая шкала)', fontsize=14)
plt.ylabel('Время выполнения (логарифмическая шкала)', fontsize=14)
plt.title('Зависимость времени от размера матрицы\n(логарифмическая шкала)', fontsize=16)
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('results_cuda/fig3_log_scale.png', dpi=200, bbox_inches='tight')
plt.show()
print(" fig3_log_scale.png сохранён")

# ============================================
# График 4: Сравнение конфигураций (ускорение относительно 8x8)
# ============================================
plt.figure(figsize=(10, 6))

base_times = [times[i] for i in range(len(sizes)) if configs[i] == '8x8']

for config in ['16x16', '32x32']:
    config_times = [times[i] for i in range(len(sizes)) if configs[i] == config]
    speedup = [base / config for base, config in zip(base_times, config_times)]
    
    plt.plot(unique_sizes, speedup, 'o-', 
             linewidth=2.5, markersize=8, 
             label=f'{config} относительно 8×8', 
             color=colors[config])

plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Без ускорения (1.0)')
plt.xlabel('Размер матрицы (n)', fontsize=14)
plt.ylabel('Ускорение (раз)', fontsize=14)
plt.title('Ускорение относительно конфигурации 8×8', fontsize=16)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)
plt.xticks(unique_sizes, fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.savefig('results_cuda/fig4_speedup_vs_8x8.png', dpi=200, bbox_inches='tight')
plt.show()
print(" fig4_speedup_vs_8x8.png сохранён")

print(" Все графики сохранены в папку 'results_cuda/'")