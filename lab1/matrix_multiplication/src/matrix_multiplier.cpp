#include "matrix_utils.h"
#include <fstream>
#include <random>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

Matrix MatrixUtils::generateRandomMatrix(size_t size, double min_val, double max_val) {
    Matrix matrix(size, std::vector<double>(size));
    
    // Используем случайный генератор с энтропией
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

bool MatrixUtils::saveMatrixToFile(const Matrix& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    size_t size = matrix.size();
    file << size << "\n";
    
    // Устанавливаем точность для вывода
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

Matrix MatrixUtils::multiplyMatrices(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
    // Классический алгоритм умножения матриц с оптимизацией (ikj порядок)
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

Matrix MatrixUtils::multiplyMatricesParallel(const Matrix& A, const Matrix& B) {
    size_t n = A.size();  // это оставьте как size_t для размера
    Matrix C(n, std::vector<double>(n, 0.0));
    
#ifdef _OPENMP
    // Используем тип со знаком для переменных цикла
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (long long j = 0; j < static_cast<long long>(n); ++j) {
            double sum = 0.0;
            for (long long k = 0; k < static_cast<long long>(n); ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
#else
    C = multiplyMatrices(A, B);
#endif
    return C;
}
bool MatrixUtils::compareMatrices(const Matrix& C1, const Matrix& C2, double tolerance, double& max_diff) {
    size_t n = C1.size();
    if (n != C2.size()) return false;
    
    max_diff = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double diff = std::abs(C1[i][j] - C2[i][j]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

bool MatrixUtils::verifyResult(const Matrix& A, const Matrix& B, const Matrix& C, double& max_diff) {
    // Вычисляем эталонный результат с помощью последовательного алгоритма
    // (который считаем верным)
    Matrix expected = multiplyMatrices(A, B);
    
    // Сравниваем результаты
    return compareMatrices(C, expected, 1e-8, max_diff);
}

void MatrixUtils::printMatrix(const Matrix& matrix, const std::string& name) {
    if (matrix.empty()) {
        std::cout << name << " is empty" << std::endl;
        return;
    }
    
    size_t size = matrix.size();
    std::cout << name << " (" << size << "x" << size << "):" << std::endl;
    
    // Для больших матриц выводим только первые 5x5 элементов
    const size_t max_print_size = 5;
    size_t print_size = std::min(size, max_print_size);
    
    for (size_t i = 0; i < print_size; ++i) {
        for (size_t j = 0; j < print_size; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i][j] << " ";
        }
        if (size > max_print_size) std::cout << "...";
        std::cout << std::endl;
    }
    
    if (size > max_print_size) {
        std::cout << "..." << std::endl;
    }
}

PerformanceMetrics MatrixUtils::getMetrics(size_t size) {
    PerformanceMetrics metrics;
    metrics.matrix_size = size;
    metrics.memory_used = 3 * size * size * sizeof(double); // A, B, C
    metrics.verification_passed = false;
    metrics.max_difference = -1.0;
    
    Timer timer;
    
    std::cout << "  Generating matrices..." << std::endl;
    timer.reset();
    Matrix A = generateRandomMatrix(size);
    Matrix B = generateRandomMatrix(size);
    metrics.generation_time = timer.elapsed();
    
    std::cout << "  Saving to files..." << std::endl;
    saveMatrixToFile(A, "data/matrix_a.txt");
    saveMatrixToFile(B, "data/matrix_b.txt");
    
    std::cout << "  Multiplying matrices..." << std::endl;
    timer.reset();
#ifdef _OPENMP
    Matrix C = multiplyMatricesParallel(A, B);
#else
    Matrix C = multiplyMatrices(A, B);
#endif
    metrics.multiplication_time = timer.elapsed();
    
    std::cout << "  Saving result..." << std::endl;
    saveMatrixToFile(C, "data/result_matrix.txt");
    
    std::cout << "  Verifying result..." << std::endl;
    timer.reset();
    metrics.verification_passed = verifyResult(A, B, C, metrics.max_difference);
    metrics.verification_time = timer.elapsed();
    
    return metrics;
}