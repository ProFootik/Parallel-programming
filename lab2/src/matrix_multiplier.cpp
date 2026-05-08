#include "matrix_utils.h"
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

// Генерация случайной матрицы
Matrix MatrixUtils::generateRandomMatrix(size_t size, double min_val, double max_val) {
    Matrix matrix(size, std::vector<double>(size));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    
    return matrix;
}

// Сохранение матрицы в файл
bool MatrixUtils::saveMatrixToFile(const Matrix& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    size_t size = matrix.size();
    file << size << "\n";
    file << std::fixed << std::setprecision(6);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            file << matrix[i][j];
            if (j < size - 1) file << " ";
        }
        file << "\n";
    }
    
    file.close();
    return true;
}

// Загрузка матрицы из файла
Matrix MatrixUtils::loadMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
        return Matrix();
    }
    
    size_t size;
    file >> size;
    Matrix matrix(size, std::vector<double>(size));
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            file >> matrix[i][j];
        }
    }
    
    file.close();
    return matrix;
}

// Последовательное умножение матриц (оптимизированный алгоритм)
Matrix MatrixUtils::multiplyMatricesSequential(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
    // Используем оптимизированный порядок циклов (ikj) для лучшего использования кэша
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = A[i][k];
            for (size_t j = 0; j < n; ++j) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    
    return C;
}

// Параллельное умножение матриц с использованием OpenMP
Matrix MatrixUtils::multiplyMatricesParallel(const Matrix& A, const Matrix& B, 
                                               const std::string& schedule_type) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
#ifdef _OPENMP
    long long n_signed = static_cast<long long>(n);
    
    // Выбор стратегии распределения итераций
    if (schedule_type == "static") {
        // Статическое распределение - для равномерной нагрузки
        #pragma omp parallel for collapse(2) schedule(static)
        for (long long i = 0; i < n_signed; ++i) {
            for (long long j = 0; j < n_signed; ++j) {
                double sum = 0.0;
                for (long long k = 0; k < n_signed; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
    else if (schedule_type == "dynamic") {
        // Динамическое распределение с chunk_size=32 - для лучшей балансировки
        #pragma omp parallel for collapse(2) schedule(dynamic, 32)
        for (long long i = 0; i < n_signed; ++i) {
            for (long long j = 0; j < n_signed; ++j) {
                double sum = 0.0;
                for (long long k = 0; k < n_signed; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
    else { // guided
        // Guided распределение - адаптивное
        #pragma omp parallel for collapse(2) schedule(guided)
        for (long long i = 0; i < n_signed; ++i) {
            for (long long j = 0; j < n_signed; ++j) {
                double sum = 0.0;
                for (long long k = 0; k < n_signed; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
#else
    C = multiplyMatricesSequential(A, B);
#endif
    
    return C;
}

// Сравнение двух матриц с заданной погрешностью
bool MatrixUtils::compareMatrices(const Matrix& C1, const Matrix& C2, double tolerance, double& max_diff) {
    size_t n = C1.size();
    if (n != C2.size()) return false;
    
    max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double diff = std::abs(C1[i][j] - C2[i][j]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) return false;
        }
    }
    return true;
}

// Верификация результата путем сравнения с последовательной версией
bool MatrixUtils::verifyResult(const Matrix& A, const Matrix& B, const Matrix& C, double& max_diff) {
    Matrix expected = multiplyMatricesSequential(A, B);
    return compareMatrices(C, expected, 1e-8, max_diff);
}

// Вывод матрицы на экран
void MatrixUtils::printMatrix(const Matrix& matrix, const std::string& name, size_t max_print) {
    if (matrix.empty()) {
        std::cout << name << " is empty" << std::endl;
        return;
    }
    
    size_t size = matrix.size();
    std::cout << name << " (" << size << "x" << size << "):" << std::endl;
    
    size_t print_size = std::min(size, max_print);
    
    for (size_t i = 0; i < print_size; ++i) {
        for (size_t j = 0; j < print_size; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i][j] << " ";
        }
        if (size > max_print) std::cout << "...";
        std::cout << std::endl;
    }
    
    if (size > max_print) {
        std::cout << "..." << std::endl;
    }
}

// Получение метрик производительности для заданного количества потоков
PerformanceMetrics MatrixUtils::getMetricsWithThreads(size_t size, int num_threads) {
    PerformanceMetrics metrics;
    metrics.matrix_size = size;
    metrics.num_threads = num_threads;
    metrics.memory_used = 3 * size * size * sizeof(double);
    metrics.verification_passed = false;
    metrics.max_difference = -1.0;
    
    Timer timer;
    
    // Генерация матриц
    timer.reset();
    Matrix A = generateRandomMatrix(size);
    Matrix B = generateRandomMatrix(size);
    metrics.generation_time = timer.elapsed();
    
    // Умножение матриц с использованием OpenMP
    timer.reset();
    
#ifdef _OPENMP
    // Устанавливаем количество потоков
    omp_set_num_threads(num_threads);
    
    // Выполняем умножение с динамическим распределением
    Matrix C = multiplyMatricesParallel(A, B, "dynamic");
#else
    Matrix C = multiplyMatricesSequential(A, B);
#endif
    
    metrics.multiplication_time = timer.elapsed();
    
    // Верификация результата
    timer.reset();
    metrics.verification_passed = verifyResult(A, B, C, metrics.max_difference);
    metrics.verification_time = timer.elapsed();
    
    // Сохранение матриц в файлы (только для первого запуска)
    static bool saved_once = false;
    if (!saved_once) {
        // Создаем папку results если её нет
        #ifdef _WIN32
        system("if not exist results mkdir results");
        #else
        system("mkdir -p results");
        #endif
        
        saveMatrixToFile(A, "results/matrix_a.txt");
        saveMatrixToFile(B, "results/matrix_b.txt");
        saveMatrixToFile(C, "results/result_matrix.txt");
        saved_once = true;
    }
    
    return metrics;
}

// Сохранение результатов тестирования в CSV файл
void MatrixUtils::saveResultsToFile(size_t size, int threads, double time, double speedup, double efficiency) {
    // Создаем папку results если её нет
    #ifdef _WIN32
    system("if not exist results mkdir results");
    #else
    system("mkdir -p results");
    #endif
    
    std::ofstream file("results/performance_data.csv", std::ios::app);
    
    if (file.tellp() == 0) {
        // Заголовок CSV файла
        file << "MatrixSize,Threads,Time_seconds,Speedup,Efficiency_Percent\n";
    }
    
    file << size << "," << threads << "," << time << "," << speedup << "," << (efficiency * 100) << "\n";
    file.close();
}

// Прогрев кэша перед измерениями
void MatrixUtils::warmup(size_t size) {
    static bool warmed_up = false;
    if (!warmed_up) {
        size_t warm_size = std::min(size, size_t(200));
        Matrix A = generateRandomMatrix(warm_size);
        Matrix B = generateRandomMatrix(warm_size);
        Matrix C = multiplyMatricesSequential(A, B);
        warmed_up = true;
    }
}

// Расчет теоретической производительности в GFLOPS
double MatrixUtils::calculateTheoreticalFlops(size_t size, double time) {
    // Количество операций: 2 * n^3 (умножение и сложение)
    double operations = 2.0 * size * size * size;
    double gflops = operations / (time * 1e9);
    return gflops;
}