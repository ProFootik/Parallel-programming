#include "matrix_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "Matrix Multiplication Performance Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
#ifdef _OPENMP
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP not available - running sequential version" << std::endl;
#endif
    
    // Размеры матриц для тестирования
    std::vector<size_t> test_sizes;
    if (argc > 1) {
        // Можно передать размер через аргументы командной строки
        test_sizes.push_back(std::stoul(argv[1]));
    } else {
        // Тестовые размеры
        test_sizes = {100, 200, 500, 1000};
        
        // Добавляем маленькие размеры для демонстрации вывода матриц
        test_sizes.insert(test_sizes.begin(), 5);
    }
    
    std::cout << "\nTest configurations:" << std::endl;
    for (size_t size : test_sizes) {
        double memory_mb = 3.0 * size * size * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "  - Matrix size: " << size << "x" << size 
                  << " (memory: ~" << std::fixed << std::setprecision(1) << memory_mb << " MB)" 
                  << std::endl;
    }
    std::cout << std::endl;
    
    // Создаем директорию для данных
    system("mkdir -p data");
    
    // Заголовок таблицы результатов
    std::cout << std::left 
              << std::setw(12) << "Size"
              << std::setw(15) << "Gen Time (s)"
              << std::setw(15) << "Mult Time (s)"
              << std::setw(15) << "Verify Time (s)"
              << std::setw(15) << "Memory (MB)"
              << std::setw(20) << "Verification"
              << std::setw(15) << "Max Diff"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    
    // Выполняем тесты
    for (size_t size : test_sizes) {
        std::cout << std::flush;
        std::cout << "Running test for size " << size << "..." << std::endl;
        
        PerformanceMetrics metrics = MatrixUtils::getMetrics(size);
        
        // Вывод результатов
        std::cout << std::left
                  << std::setw(12) << metrics.matrix_size
                  << std::setw(15) << std::fixed << std::setprecision(6) << metrics.generation_time
                  << std::setw(15) << std::fixed << std::setprecision(6) << metrics.multiplication_time
                  << std::setw(15) << std::fixed << std::setprecision(6) << metrics.verification_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << (metrics.memory_used / (1024.0 * 1024.0))
                  << std::setw(20) << (metrics.verification_passed ? "✓ PASSED" : "✗ FAILED")
                  << std::setw(15) << std::scientific << std::setprecision(2) << metrics.max_difference
                  << std::endl;
        
        // Для небольших матриц выводим их содержимое
        if (size <= 10) {
            Matrix A = MatrixUtils::loadMatrixFromFile("data/matrix_a.txt");
            Matrix B = MatrixUtils::loadMatrixFromFile("data/matrix_b.txt");
            Matrix C = MatrixUtils::loadMatrixFromFile("data/result_matrix.txt");
            
            MatrixUtils::printMatrix(A, "\nMatrix A");
            MatrixUtils::printMatrix(B, "Matrix B");
            MatrixUtils::printMatrix(C, "Result Matrix C");
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "\nResults saved to:" << std::endl;
    std::cout << "  - data/matrix_a.txt" << std::endl;
    std::cout << "  - data/matrix_b.txt" << std::endl;
    std::cout << "  - data/result_matrix.txt" << std::endl;
    
    std::cout << "\nVerification method: Comparison with sequential algorithm" << std::endl;
    std::cout << "Tolerance: 1e-8" << std::endl;
    
    return 0;
}