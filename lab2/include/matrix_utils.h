#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// Тип матрицы
using Matrix = std::vector<std::vector<double>>;

// Структура для хранения метрик производительности
struct PerformanceMetrics {
    size_t matrix_size;
    int num_threads;
    double generation_time;
    double multiplication_time;
    double verification_time;
    size_t memory_used;
    double max_difference;
    bool verification_passed;
};

// Основной класс для работы с матрицами
class MatrixUtils {
public:
    // Генерация случайной матрицы
    static Matrix generateRandomMatrix(size_t size, double min_val = -10.0, double max_val = 10.0);
    
    // Сохранение и загрузка матриц
    static bool saveMatrixToFile(const Matrix& matrix, const std::string& filename);
    static Matrix loadMatrixFromFile(const std::string& filename);
    
    // Умножение матриц (различные версии)
    static Matrix multiplyMatricesSequential(const Matrix& A, const Matrix& B);
    static Matrix multiplyMatricesParallel(const Matrix& A, const Matrix& B, 
                                           const std::string& schedule_type = "dynamic");
    
    // Верификация результатов
    static bool verifyResult(const Matrix& A, const Matrix& B, const Matrix& C, double& max_diff);
    static bool compareMatrices(const Matrix& C1, const Matrix& C2, double tolerance, double& max_diff);
    
    // Вывод матрицы на экран
    static void printMatrix(const Matrix& matrix, const std::string& name, size_t max_print = 5);
    
    // Получение метрик производительности
    static PerformanceMetrics getMetricsWithThreads(size_t size, int num_threads);
    
    // Сохранение результатов в файл
    static void saveResultsToFile(size_t size, int threads, double time, double speedup, double efficiency);
    
    // Прогрев кэша перед измерениями
    static void warmup(size_t size);
    
    // Расчет теоретической производительности
    static double calculateTheoreticalFlops(size_t size, double time);
};

// Класс для измерения времени
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};

#endif // MATRIX_UTILS_H