#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

// Типы матриц
using Matrix = std::vector<std::vector<double>>;

// Структура для хранения результатов измерений
struct PerformanceMetrics {
    size_t matrix_size;
    double generation_time;
    double multiplication_time;
    double verification_time;
    size_t memory_used; // в байтах
    double max_difference; // максимальная разница при верификации
    bool verification_passed;
};

// Функции для работы с матрицами
class MatrixUtils {
public:
    // Генерация случайной матрицы
    static Matrix generateRandomMatrix(size_t size, double min_val = -10.0, double max_val = 10.0);
    
    // Сохранение матрицы в файл
    static bool saveMatrixToFile(const Matrix& matrix, const std::string& filename);
    
    // Загрузка матрицы из файла
    static Matrix loadMatrixFromFile(const std::string& filename);
    
    // Умножение матриц (последовательная версия)
    static Matrix multiplyMatrices(const Matrix& A, const Matrix& B);
    
    // Умножение матриц (параллельная версия с OpenMP)
    static Matrix multiplyMatricesParallel(const Matrix& A, const Matrix& B);
    
    // Верификация результатов (сравнение с эталонным умножением)
    static bool verifyResult(const Matrix& A, const Matrix& B, const Matrix& C, double& max_diff);
    
    // Вывод матрицы на экран (для небольших размеров)
    static void printMatrix(const Matrix& matrix, const std::string& name);
    
    // Получение метрик производительности
    static PerformanceMetrics getMetrics(size_t size);
    
    // Сравнение двух матриц с погрешностью
    static bool compareMatrices(const Matrix& C1, const Matrix& C2, double tolerance, double& max_diff);
};

// Вспомогательные функции для работы со временем
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