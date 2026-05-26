ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ

РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:

ТАБЛИЦА:
<img width="856" height="462" alt="image" src="https://github.com/user-attachments/assets/bae82757-ca24-4bff-bb79-32e8a267f65d" />

ГРАФИК(ось х - размер матрицы, ось у - время умножения):
<img width="778" height="472" alt="image" src="https://github.com/user-attachments/assets/21a2aa9f-f6ad-4ee5-98de-29ef9b1da666" />

Анализ производительности

Наблюдения:

Время умножения растет пропорционально O(n³) - кубическая зависимость

Для матрицы 1000×1000 время умножения составило 8.73 секунды

Верификация занимает примерно в 1.5 раза больше времени, чем само умножение

Максимальная погрешность вычислений: 0.00e+00 (в пределах машинной точности)

Метрики:

Ускорение при использовании 4 потоков: ~3.2x (по сравнению с последовательной версией)

Эффективность параллелизации: ~80%

Объем вычислений для n=1000: 1×10⁹ операций

Выводы

1. Корректность: Разработанная программа правильно выполняет умножение матриц, что подтверждено автоматической верификацией для всех тестовых размеров.

2. Производительность:

Использование OpenMP с 4 потоками дает ускорение ~3.2 раза

Наибольшее время (8.73 с) получено для максимального размера 1000×1000

Время выполнения соответствует теоретической сложности O(n³)

3. Масштабируемость:

Для малых матриц (n<200) накладные расходы на параллелизацию значительны

Для больших матриц (n≥500) параллельная версия эффективно использует все ядра

4. Использование памяти: Программа эффективно использует память, объем точно соответствует теоретическим расчетам.

ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ 2


Цель работы:
Исследовать эффективность параллельного умножения квадратных матриц с использованием технологии OpenMP в зависимости от:

-Количества потоков (1, 2, 4, 8)

-Размера матриц (200, 400, 800, 1200, 1600, 2000)

-Оценка ускорения и эффективности параллелизации

Таблица результатов:

<img width="780" height="583" alt="image" src="https://github.com/user-attachments/assets/00a9c577-cea2-48f0-9cc2-f201b09da4a0" />

График зависимости времени от размерности:

<img width="846" height="572" alt="image" src="https://github.com/user-attachments/assets/3efcae0f-79f9-49ed-9c67-71bb5e8f8827" />

График зависимоти ускорения от количества потоков:

<img width="844" height="704" alt="image" src="https://github.com/user-attachments/assets/8a07270a-4506-4830-9308-d7ec2ff19f5a" />




АНАЛИЗ РЕЗУЛЬТАТОВ

Наблюдения:

1. Максимальное ускорение (3.64x) достигнуто для матрицы 1200×1200 при 8 потоках

2. Оптимальное количество потоков - 4 (физические ядра)

3. При использовании 8 потоков ускорение растет незначительно из-за Hyper-Threading



ВЫВОДЫ

1. Эффективность OpenMP:

- Достигнуто ускорение до 3.64x на 8 потоках

- Оптимальное ускорение 3.42x на 4 потоках

- Эффективность параллелизации достигает 85.5% для матриц 1200×1200

2. Масштабируемость:

- Хорошая масштабируемость до 4 потоков (физические ядра)

- Дальнейшее увеличение потоков неэффективно из-за Hyper-Threading

- Доля параллельного кода составляет ~94.4%

3. Влияние размера задачи:

- Минимальный эффективный размер: 400×400

- Оптимальный диапазон: 800×800 - 1600×1600

- Для малых матриц накладные расходы доминируют




Отчёт по лабораторной работе: Перемножение матриц на C++ с использованием технологии MPI
1. Задание
Модифицировать программу из л/р №1 для параллельной работы по технологии MPI. Провести серию экспериментов с разными размерами матриц (примерно 200, 400, 800, 1200, 1600, 2000), с разным количеством вычислительных ядер (1, 2, 4, 8 и т.д.).

2. Описание работы скриптов
2.1. generate_matrices.py
Использует numpy.random.randint для генерации целочисленных значений.
Порядок матрицы n_matrix задан внутри скрипта (например, n_matrix = 1000).
Записывает n_matrix в первую строку текстового файла, затем сгенерированный элементы матрицы построчно.
2.2. verification_matrix.py
Загружает матрицы A, B и результат C++ из файлов по фиксированным путям.
Считывает порядок квадратных матриц n_matrix из первой строки каждого файла.
Вычисляет эталонное произведение через np.dot и выполняет точное поэлементное сравнение с результатом C++ с помощью np.array_equal.
При успешном совпадении сохраняет эталонную матрицу в файл verification_result_C.txt для возможности визуального анализа и выводит подтверждение в консоль.
В случае ошибки выводит уведомление и завершает работу.
2.3. matrix_multiplication.cpp
Матрицы считываются из текстовых файлов и хранятся в vector<long long> как одномерные массивы для повышения локальности данных.
Доступ к элементу (i, j) осуществляется по формуле i * N + j, где N — порядок квадратной матрицы.
Параллелизация вычислений реализована средствами MPI: главный процесс (rank 0) распределяет строки матрицы A между процессами через MPI_Scatterv, рассылает матрицу B целиком через MPI_Bcast и собирает итоговые части результата через MPI_Gatherv.
Вычисление элементов результирующей матрицы производится по классическому алгоритму с порядком обхода циклов i -> j -> k.
Измерение времени выполнения программы осуществляется с помощью chrono::high_resolution_clock.
Результаты замеров (время в мс) и итоговая рассчитанная матрица записываются в result_C.txt.
2.4. Код matrix_multiplication.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using namespace std;

vector<long long> read_matrix(const string& filename, size_t& size)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << endl;
        exit(1);
    }

    file >> size;

    vector<long long> matrix_value(size * size);
    for (int i = 0; i < size * size; ++i)
    {
        if (!(file >> matrix_value[i]))
        {
            std::cerr << "Ошибка: Недостаточно данных в файле " << filename << " для " << size << "x" << size << " матриц" << endl;
            exit(1);
        }
    }
    return matrix_value;
}

void write_matrix(const string& filename, const vector<long long>& matrix_flat, size_t& size, auto& duration)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Ошибка открытия файла:  " << filename << endl;
        exit(1);
    }

    file << size << endl;
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            file << matrix_flat[i * size + j] << (j == size - 1 ? "" : " ");
        }

        file << endl;
    }

    file << "Время выполнения задачи: " << duration.count() << " ms" << endl;
    file << "Объём задачи: " << size << " - порядок матриц";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    string file_A = "C:\\Users\\Адель\\Desktop\\lab-1\\lab_3\\parallel-programming\\Data_for_size=2000\\matrix_A.txt";
    string file_B = "C:\\Users\\Адель\\Desktop\\lab-1\\lab_3\\parallel-programming\\Data_for_size=2000\\matrix_B.txt";
    string result_file = "C:\\Users\\Адель\\Desktop\\lab-1\\lab_3\\result_C.txt";

    size_t N = 0;
    vector<long long> A_matrix_values;
    vector<long long> B_matrix_values;
    vector<long long> result_matrix;

    if (rank == 0)
    {
        size_t A_matrix_size, B_matrix_size;
        A_matrix_values = read_matrix(file_A, A_matrix_size);
        B_matrix_values = read_matrix(file_B, B_matrix_size);

        if (A_matrix_size != B_matrix_size)
        {
            cerr << "Ошибка: Размеры матрицы не совпадают" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        N = A_matrix_size;
        result_matrix.resize(N * N, 0LL);
    }

    unsigned long long N_ull = N;
    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_ull);

    if (rank != 0)
    {
        B_matrix_values.resize(N * N);
    }

    MPI_Bcast(B_matrix_values.data(), N * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size_mpi;
    int remainder = N % size_mpi;

    vector<int> sendcounts(size_mpi);
    vector<int> displs(size_mpi);

    int offset = 0;
    for (int i = 0; i < size_mpi; ++i)
    {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_elements = sendcounts[rank];
    int local_rows = local_elements / N;

    vector<long long> local_A(local_elements);
    vector<long long> local_C(local_elements, 0LL);

    chrono::time_point<chrono::high_resolution_clock> start_time;
    if (rank == 0)
    {
        start_time = chrono::high_resolution_clock::now();
    }

    MPI_Scatterv(rank == 0 ? A_matrix_values.data() : nullptr, sendcounts.data(), displs.data(), MPI_LONG_LONG,
        local_A.data(), local_elements, MPI_LONG_LONG,
        0, MPI_COMM_WORLD);

    for (size_t i = 0; i < local_rows; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            long long sum = 0LL;
            for (size_t k = 0; k < N; ++k)
            {
                sum += local_A[i * N + k] * B_matrix_values[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    MPI_Gatherv(local_C.data(), local_elements, MPI_LONG_LONG,
        rank == 0 ? result_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_LONG_LONG,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end_time - start_time;

        cout << "Объём задачи: " << N << endl;
        cout << "Время выполнения задачи (миллисекунды): " << duration.count() << endl;
        cout << "Результаты записаны в файл : " << result_file << endl;

        write_matrix(result_file, result_matrix, N, duration);
    }

    MPI_Finalize();
    return 0;
}
3. Результаты экспериментов
Для демонстрации работы программы были проведены эксперименты с разным количеством вычислительных ядер (1, 2, 3, 4, 5, 6, 7, 8) для матриц порядка 200, 400, 800, 1200, 1600, 2000.

3.1. При порядке равном 200
<img width="2451" height="1358" alt="image" src="https://github.com/user-attachments/assets/12972c5c-ecb9-4119-96af-5f94c9db49c1" />

Зависимость времени от потоков для N=200

На начальном этапе (1-4 ядра) наблюдается стабильное снижение времени до минимума в 1,92 мс. На этом отрезке выигрыш от распараллеливания превышает затраты на организацию MPI-процессов.
На 5-6 ядрах наблюдается скачок времени выполнения вверх. Это связано с дисбалансом нагрузки, так как 200 не делится нацело на 5 или 6, строки распределяются неравномерно, поэтому один из процессов работает дольше остальных, задерживая общую сборку результата.
На 8-ми ядрах время снова снижается (до 1,72 мс) благодаря идеальной кратности. Нагрузка распределяется поровну (200 / 8 = 25 строк на процесс), что минимизирует простои и оптимизирует коллективные операции MPI.
3.2. При порядке равном 400
<img width="2561" height="1503" alt="image" src="https://github.com/user-attachments/assets/498c2137-872d-4db7-a500-14f981b33372" />

Зависимость времени от потоков для N=400

На 1-7 ядрах наблюдается устойчивое снижение времени выполнения с 47,5 мс до 9 мс. Увеличение вычислительной сложности задачи позволило системе более эффективно использовать параллельные ресурсы.
При переходе к 8 ядрам происходит рост времени выполнения до 12,5 мс. Несмотря на идеальную кратность (400 / 8 = 50 строк на процесс), на данном этапе накладные расходы на синхронизацию и сборку данных начинают превалировать над выигрышем от дробления задачи.
3.3. При порядке равном 800
<img width="2183" height="1274" alt="image" src="https://github.com/user-attachments/assets/d70c5d35-b8bf-4513-b587-9dc0c6fe9191" />

Зависимость времени от потоков для N=800

Время выполнения резко сокращается с 353 мс до 116 мс. В этом диапазоне ресурсов параллельное выполнение эффективно, так как объем вычислений значительно превышает затраты на синхронизацию потоков.
После достижения локального минимума на 4 ядрах начинается постепенная деградация производительности. Время выполнения на 8 ядрах увеличивается до 136 мс.
3.4. При порядке равном 1200
<img width="2517" height="1409" alt="image" src="https://github.com/user-attachments/assets/5c3955c5-34d3-4dce-950e-befd3f43c602" />

Зависимость времени от потоков для N=1200

На 1-6 ядрах наблюдается наиболее эффективный участок снижения времени выполнения с 1314 мс до 434 мс. Рост объема данных позволяет равномерно загрузить ядра, сводя к минимуму влияние системных прерываний и задержек инициализации.
После достижения минимума (434 мс) на 6 ядрах время выполнения начинает расти и на 8 ядрах составляет 445 мс.
3.5. При порядке равном 1600
<img width="2428" height="1446" alt="image" src="https://github.com/user-attachments/assets/7c9b0e07-06c2-4e86-9fbd-64319c4f0db8" />

Зависимость времени от потоков для N=1600

На 1-8 ядрах время выполнения сокращается наиболее интенсивно с 10,7 с до 2,27 с. Большой объем обрабатываемых данных позволяет процессорам работать с максимальной загрузкой, при этом доля времени на межпроцессорное взаимодействие остается незначительной относительно полезных вычислений.
3.6. При порядке равном 2000
<img width="2428" height="1446" alt="image" src="https://github.com/user-attachments/assets/bb7e8ba6-8179-4f6f-8f9d-b6ce8ceb3f68" />


Зависимость времени от потоков для N=2000

Наблюдается плавное и монотонное снижение времени выполнения во всем диапазоне с 12,8 с на одном ядре до 3 с на восьми ядрах.
Это свидетельствует о том, что вычислительная нагрузка стала достаточно велика, чтобы полностью нивелировать влияние системных шумов и задержек связи.
4. Выводы
Параллельные вычисления демонстрируют наибольшую эффективность на матрицах высокого порядка (N >= 1600), где высокая вычислительная сложность задачи полностью оправдывает затраты на межпроцессорное взаимодействие.
Для малых и средних размерностей матриц (порядка от 200 до 800) оптимальным является использование 4–6 ядер. Дальнейшее увеличение числа процессов приводит к деградации производительности из-за дисбаланса нагрузки при некратном распределении строк.
При больших значениях N, например, N = 2000, наблюдается наиболее стабильное и предсказуемое ускорение. В этом режиме алгоритм эффективно использует все доступные вычислительные узлы, минимизируя влияние системных шумов и задержек синхронизации.







# Лабораторная работа №4: Параллельное умножение матриц с использованием CUDA

## Описание

Модификация программы последовательного умножения матриц (лаб. №1) для параллельного выполнения на GPU с использованием технологии NVIDIA CUDA. Исследование влияния конфигураций блоков на производительность.

## Конфигурация тестовой системы

- **Среда выполнения**: Google Colab GPU  
- **Видеокарта**: NVIDIA Tesla T4

## Методика экспериментов

| Параметр | Значение |
|----------|----------|
| Размеры матриц | 200, 400, 800, 1200, 1600, 2000 |
| Конфигурации блоков | 8×8, 16×16, 32×32 |
| Количество тестов | 18 (6 размеров × 3 конфигурации) |
| Верификация | Сравнение с результатами NumPy |

## Результаты экспериментов

### Таблица 1. Время выполнения (сек) и производительность (GFLOPS)

| Размер | Конфигурация | Время (сек) | GFLOPS |
|--------|--------------|-------------|--------|
| 200 | 8×8 | 0.097045 | 0.16 |
| 200 | **16×16** | **0.000125** | **128.24** |
| 200 | 32×32 | 0.000213 | 75.22 |
| 400 | 8×8 | 0.000829 | 154.40 |
| 400 | **16×16** | **0.000818** | **156.56** |
| 400 | 32×32 | 0.000925 | 138.35 |
| 800 | 8×8 | 0.006428 | 159.30 |
| 800 | **16×16** | **0.006343** | **161.43** |
| 800 | 32×32 | 0.006468 | 158.31 |
| 1200 | 8×8 | 0.038706 | 89.29 |
| 1200 | **16×16** | **0.038314** | **90.20** |
| 1200 | 32×32 | 0.039256 | 88.04 |
| 1600 | 8×8 | 0.091588 | 89.44 |
| 1600 | **16×16** | **0.072084** | **113.65** |
| 1600 | 32×32 | 0.091558 | 89.47 |
| 2000 | 8×8 | 0.178890 | 89.44 |
| 2000 | **16×16** | **0.157206** | **101.78** |
| 2000 | 32×32 | 0.159977 | 100.01 |

**Жирным** выделена лучшая конфигурация для каждого размера.

### Таблица 2. Оптимальная конфигурация и максимальная производительность

| Размер | Лучшая конфигурация | Время (сек) | GFLOPS |
|--------|---------------------|-------------|--------|
| 200 | 16×16 | 0.000125 | 128.24 |
| 400 | 16×16 | 0.000818 | 156.56 |
| 800 | 16×16 | 0.006343 | 161.43 |
| 1200 | 16×16 | 0.038314 | 90.20 |
| 1600 | 16×16 | 0.072084 | 113.65 |
| 2000 | 16×16 | 0.157206 | 101.78 |

## Графики

| Зависимость | График |
|-------------|--------|
| Время выполнения от размера матрицы | `fig1_time_vs_size` |
<img width="1982" height="1183" alt="image" src="https://github.com/user-attachments/assets/951423e0-2b0e-4778-b43b-bcec6fd21819" />

| Производительность от размера матрицы | `fig2_gflops_vs_size` |
<img width="1982" height="1183" alt="image" src="https://github.com/user-attachments/assets/7ddf8043-b73c-439c-a2df-79afc532947c" />

| Логарифмическая шкала времени | `fig3_log_scale` |
<img width="1980" height="1172" alt="image" src="https://github.com/user-attachments/assets/3dd02c44-cef4-4700-afc0-5d723f6cd27f" />

| Ускорение относительно конфигурации 8×8 | `fig4_speedup_vs_8x8` |
<img width="1982" height="1184" alt="image" src="https://github.com/user-attachments/assets/edee00e6-9513-465c-ae83-4d7a79f91892" />


## Верификация

Все 18 экспериментов успешно прошли автоматическую проверку:

- ✅ Сравнение с эталонными результатами NumPy  
- ✅ Максимальная абсолютная ошибка: < 1e-6  
- ✅ Статус: **CORRECT** для всех тестов

## Анализ результатов

### 1. Влияние конфигурации блоков

- **16×16** — абсолютный лидер на всех размерах матриц  
- Конфигурация **8×8** показала аномально низкую производительность на 200×200 из-за накладных расходов  
- Конфигурация **32×32** немного уступает 16×16 на всех размерах

### 2. Зависимость от размера матрицы

- **Пиковая производительность**: 161.43 GFLOPS (800×800, 16×16)  
- Снижение GFLOPS на больших матрицах (>1200) связано с промахами кэша GPU  
- **Минимальное время** для 2000×2000: 0.157 сек (16×16)

### 3. Эффективность CUDA

| Показатель | Значение |
|------------|----------|
| Максимальная производительность | 161.43 GFLOPS |
| Теоретический максимум Tesla T4 (FP64) | ~250 GFLOPS |
| **Эффективность** | **~64% от пика** |

## Запуск программы

```bash
python generate_matrices.py && \
nvcc -o matrix_multiplication_cuda matrix_multiplication_cuda.cu && \
./matrix_multiplication_cuda && \
python verify_results.py && \
python plot_graphs.py
```

Выводы
Программа успешно модифицирована для параллельного выполнения на GPU с использованием CUDA

Корректность подтверждена — все 18 тестов прошли верификацию

Оптимальная конфигурация блоков — 16×16 для всех исследованных размеров матриц (200–2000)

Достигнута производительность 161.43 GFLOPS на матрицах 800×800, что составляет ~64% от теоретического максимума Tesla T4

Накладные расходы GPU заметны только для маленьких матриц (200×200) при использовании конфигурации 8×8

Масштабируемость: время выполнения растёт пропорционально O(n³), что соответствует теоретической сложности умножения матриц





# Отчёт: Лабораторная работа №5.  Умножение двух квадратных матриц на C++ + MPI

## Цель работы

Запустить программу из ЛР №3 для параллельной работы по технологии MPI на суперкомпьютере **«Сергей Королев»**.

---

# 1. Описание решения

## Программа на C++

### Входные данные

Два файла:

- `matrixA_N.txt`
- `matrixB_N.txt`

Матрицы хранятся:
- построчно;
- элементы разделены пробелами;
- без заголовка с размерностью `N`.

### Выходные данные

Файл:
- `result_N.txt`

Также в консоль выводятся:
- время выполнения умножения (в секундах, с точностью 4 знака);
- размер матрицы `N × N`;
- объём задачи (`N³` операций);
- количество MPI-процессов.

---

## Компиляция

```bash
mpicxx matrix_mul.cpp -o matrix_mul
```

## Пример запуска через sbatch

```bash
sbatch --ntasks=4 --ntasks-per-node=4 --time=0:05:00 --partition=batch \
--wrap="module load intel/mpi4 && mpirun -n 4 ./matrix_mul"
```

---

# 2. Автоматизированная верификация

Отдельный скрипт `verify.py`:

- загружает матрицы через NumPy;
- вычисляет `A @ B`;
- сравнивает результат с программой C++;
- использует:

```python
np.allclose(..., atol=1e-5)
```

Если расхождение превышает `1e-5`, выводится максимальная ошибка.

---

# 3. Эксперименты

Размеры матриц:

```text
200, 400, 800, 1200, 1600, 2000
```

Для каждого размера:

1. Генерируются случайные матрицы;
2. Запускается `./matrix_mul`;
3. Замеряется время выполнения;
4. Выполняется верификация результата.

---

## Автоматизация экспериментов

Скрипт `run_experiments.py` автоматически:

- запускает все тесты;
- сохраняет результаты в `results_mpi.csv`;
- строит график `time_vs_n_mpi.png`.

---

# 4. Файлы проекта

## `matrix_mul.cpp`

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

using namespace std;

vector<vector<double>> read_matrix(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> mat;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        vector<double> row;
        double val;

        while (ss >> val)
            row.push_back(val);

        if (!row.empty())
            mat.push_back(row);
    }

    return mat;
}

bool is_square(const vector<vector<double>>& m) {
    if (m.empty()) return false;

    size_t n = m.size();

    return all_of(
        m.begin(),
        m.end(),
        [n](const auto& r) {
            return r.size() == n;
        }
    );
}

void write_matrix(
    const string& filename,
    const vector<vector<double>>& mat
) {
    ofstream file(filename);

    for (const auto& row : mat) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << fixed
                 << setprecision(6)
                 << row[i];

            if (i < row.size() - 1)
                file << " ";
        }

        file << endl;
    }
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) {

        if (rank == 0) {
            cerr << "Использование: "
                 << "mpirun -np <num> "
                 << argv[0]
                 << " <matrixA> <matrixB> <result>"
                 << endl;
        }

        MPI_Finalize();
        return 1;
    }

    vector<vector<double>> full_A, full_B;

    int n = 0;

    if (rank == 0) {

        full_A = read_matrix(argv[1]);
        full_B = read_matrix(argv[2]);

        if (
            !is_square(full_A) ||
            !is_square(full_B) ||
            full_A.size() != full_B.size()
        ) {

            cerr << "Ошибка: "
                 << "матрицы не квадратные "
                 << "или разных размеров!"
                 << endl;

            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        n = static_cast<int>(full_A.size());
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    vector<double> B_flat(n * n, 0.0);

    if (rank == 0) {

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                B_flat[i * n + j] = full_B[i][j];
    }

    MPI_Bcast(
        B_flat.data(),
        n * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    vector<vector<double>> B(
        n,
        vector<double>(n)
    );

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = B_flat[i * n + j];

    int rows_per_proc_base = n / world_size;
    int remainder = n % world_size;

    int local_rows =
        rows_per_proc_base +
        (rank < remainder ? 1 : 0);

    vector<int> sendcounts(world_size);
    vector<int> displs(world_size, 0);

    int current_displ = 0;

    for (int i = 0; i < world_size; ++i) {

        int proc_rows =
            rows_per_proc_base +
            (i < remainder ? 1 : 0);

        sendcounts[i] = proc_rows * n;

        displs[i] = current_displ;

        current_displ += sendcounts[i];
    }

    vector<double> A_flat(n * n, 0.0);

    if (rank == 0) {

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A_flat[i * n + j] = full_A[i][j];
    }

    vector<double> A_local_flat(local_rows * n);

    MPI_Scatterv(
        A_flat.data(),
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        A_local_flat.data(),
        local_rows * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    vector<vector<double>> A_local(
        local_rows,
        vector<double>(n)
    );

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            A_local[i][j] =
                A_local_flat[i * n + j];

    vector<vector<double>> C_local(
        local_rows,
        vector<double>(n, 0.0)
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = 0.0;

    if (rank == 0)
        start_time = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C_local[i][j] +=
                    A_local[i][k] * B[k][j];

    vector<double> C_local_flat(
        local_rows * n,
        0.0
    );

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            C_local_flat[i * n + j] =
                C_local[i][j];

    vector<int> recvcounts = sendcounts;
    vector<int> recvdispls = displs;

    vector<double> C_flat(n * n, 0.0);

    MPI_Gatherv(
        C_local_flat.data(),
        local_rows * n,
        MPI_DOUBLE,
        C_flat.data(),
        recvcounts.data(),
        recvdispls.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        double duration =
            MPI_Wtime() - start_time;

        vector<vector<double>> C(
            n,
            vector<double>(n)
        );

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] =
                    C_flat[i * n + j];

        write_matrix(argv[3], C);

        long long volume =
            (long long)n * n * n;

        cout << "Время выполнения: "
             << fixed
             << setprecision(4)
             << duration
             << " секунд"
             << endl;

        cout << "Количество процессов: "
             << world_size
             << endl;

        cout << "Размер матрицы: "
             << n
             << " x "
             << n
             << endl;

        cout << "Объем задачи: "
             << volume
             << " операций"
             << endl;
    }

    MPI_Finalize();
    return 0;
}
```

---

## `generate_matrices.py`

```python
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
```

---

## `run_experiments.py`

```python
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

            proc = subprocess.run(
                [
                    "mpirun",
                    "-np",
                    str(np_val),
                    "--oversubscribe",
                    "./matrix_mul",
                    a_file,
                    b_file,
                    res_file
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            output = proc.stdout + proc.stderr

            time_match = re.search(
                r"Время выполнения: ([\d.]+) секунд",
                output
            )

            if time_match:

                t = float(time_match.group(1))

                vol = n * n * n

                results.append({
                    "N": n,
                    "NP": np_val,
                    "Time_s": t,
                    "Volume": vol
                })

                print(
                    f"N={n:4d} | NP={np_val:2d} "
                    f"| Время: {t:8.4f} с"
                )

            else:

                print(
                    f"N={n} NP={np_val} "
                    f"— не удалось прочитать время"
                )

                if output.strip():
                    print("Вывод программы:")
                    print(output.strip()[:800])

        except Exception as e:
            print(f"N={n} NP={np_val} — ошибка: {e}")

df = pd.DataFrame(results)

df.to_csv("results_mpi.csv", index=False)

print("\nРезультаты сохранены в results_mpi.csv")

plt.figure(figsize=(12, 7))

for np_val in sorted(df["NP"].unique()):

    sub = df[df["NP"] == np_val]

    plt.plot(
        sub["N"],
        sub["Time_s"],
        marker='o',
        linewidth=2,
        label=f"{np_val} процессов"
    )

plt.xlabel("Размер матрицы N")
plt.ylabel("Время выполнения (секунды)")
plt.title("Умножение матриц MPI (наивный алгоритм)")

plt.grid(True)
plt.legend()

plt.savefig("time_vs_n_mpi.png")
plt.show()

print("График сохранён: time_vs_n_mpi.png")
```

---

## `verify.py`

```python
import numpy as np
import sys

if len(sys.argv) != 4:

    print(
        "Использование: "
        "python verify.py "
        "matrixA.txt matrixB.txt result.txt"
    )

    sys.exit(1)

A = np.loadtxt(sys.argv[1])
B = np.loadtxt(sys.argv[2])
C_cpp = np.loadtxt(sys.argv[3])

C_py = A @ B

if np.allclose(C_cpp, C_py, atol=1e-5):

    print(
        "Верификация успешна! "
        "Результаты совпадают."
    )

else:

    max_diff = np.max(np.abs(C_cpp - C_py))

    print("Верификация не пройдена!")

    print(
        f"Максимальное расхождение: "
        f"{max_diff}"
    )
```

---

## `MPI.pbs`

```bash
#!/bin/bash
#SBATCH --job-name=matrix_mul
#SBATCH --time=0:05:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch

module load intel/mpi4

mpirun -r ssh ./matrix_mul
```

---

# 5. Запуск на суперкомпьютере

```bash
mpicxx matrix_mul.cpp -o matrix_mul
```

```bash
sbatch MPI.pbs
```

---

# 6. Результаты экспериментов

## Таблица времени выполнения

| Размер матрицы | NP=1 (с) | NP=2 (с) | NP=4 (с) | NP=8 (с) | Кол-во операций |
|---|---:|---:|---:|---:|---:|
| 200  | 0.0813 | 0.0453 | 0.0215 | 0.0103 | 8000000 |
| 400  | 0.6935 | 0.3509 | 0.4358 | 0.1735 | 64000000 |
| 800  | 5.1571 | 2.1654 | 1.8036 | 0.9156 | 512000000 |
| 1200 | 17.6876 | 10.6146 | 5.9585 | 2.8292 | 1728000000 |
| 1600 | 54.4936 | 28.7965 | 14.4045 | 7.2385 | 4096000000 |
| 2000 | 57.4872 | 39.0341 | 27.3586 | 12.1357 | 8000000000 |

---

## Эффективность параллелизации

| N | NP=2 | NP=4 | NP=8 |
|---|---:|---:|---:|
| 200  | 0.90 | 0.95 | 0.99 |
| 400  | 0.99 | 0.95 | 0.50 |
| 800  | 1.19 | 0.72 | 0.70 |
| 1200 | 0.84 | 0.74 | 0.78 |
| 1600 | 0.95 | 0.95 | 0.94 |
| 2000 | 0.74 | 0.53 | 0.59 |

---

# 7. График

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/6b64aa2f-d2b4-41a4-b905-4143a11688f0" />


---

# 8. Вывод

Параллельная реализация MPI показала высокую эффективность при работе с большими матрицами (`N ≥ 800`).

Для размера `N = 1600` ускорение на 8 процессах составило примерно:

```text
S ≈ 7.5
```

а эффективность параллелизации достигла:

```text
E ≈ 94%
```

что близко к идеальному масштабированию.

Для небольших размеров матриц (`N ≤ 400`) влияние коммуникационных затрат и накладных расходов MPI становится заметным, поэтому ускорение ограничено размером задачи.

Средняя вычислительная производительность на один процесс составляет примерно:

```text
200 MFlops
```
