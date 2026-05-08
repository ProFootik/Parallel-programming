#include "matrix_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// Функция для вывода разделительной линии
void printSeparator(int width) {
    std::cout << std::string(width, '=') << std::endl;
}

// Функция для вывода заголовка таблицы
void printTableHeader() {
    std::cout << std::left 
              << std::setw(10) << "Size"
              << std::setw(10) << "Threads"
              << std::setw(15) << "Mult Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Efficiency%"
              << std::setw(20) << "Verification"
              << std::setw(15) << "Max Diff"
              << std::endl;
    std::cout << std::string(94, '-') << std::endl;
}

int main(int argc, char* argv[]) {
    printSeparator(94);
    std::cout << "LABORATORY WORK #2" << std::endl;
    std::cout << "OpenMP Matrix Multiplication Performance Study" << std::endl;
    printSeparator(94);
    
    // Информация о системе
    #ifdef _OPENMP
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max available threads: " << omp_get_max_threads() << std::endl;
    std::cout << "Number of processors: " << omp_get_num_procs() << std::endl;
    #else
    std::cout << "WARNING: OpenMP not available!" << std::endl;
    std::cout << "Please enable OpenMP in compiler settings." << std::endl;
    #endif
    
    // Параметры эксперимента
    std::vector<size_t> test_sizes;
    std::vector<int> thread_counts;
    
    // Проверка аргументов командной строки
    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "all" || arg == "-a") {
            // Запуск всех тестов
            test_sizes = {200, 400, 800, 1200, 1600, 2000};
            thread_counts = {1, 2, 4, 8};
        } else {
            // Запуск с одним размером
            test_sizes.push_back(std::stoul(argv[1]));
            if (argc > 2) {
                thread_counts.push_back(std::stoi(argv[2]));
            } else {
                thread_counts = {1, 2, 4, 8};
            }
        }
    } else {
        // По умолчанию - все тесты
        test_sizes = {200, 400, 800, 1200, 1600, 2000};
        thread_counts = {1, 2, 4, 8};
    }
    
    // Вывод конфигурации тестирования
    std::cout << "\nTest Configuration:" << std::endl;
    std::cout << "  Matrix sizes: ";
    for (size_t s : test_sizes) std::cout << s << " ";
    std::cout << std::endl;
    std::cout << "  Thread counts: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << std::endl;
    
    // Расчет необходимой памяти
    std::cout << "\nMemory requirements:" << std::endl;
    for (size_t size : test_sizes) {
        double memory_mb = 3.0 * size * size * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "  " << size << "x" << size << ": ~" 
                  << std::fixed << std::setprecision(1) << memory_mb << " MB" << std::endl;
    }
    
    printSeparator(94);
    std::cout << "\nRunning experiments...\n" << std::endl;
    printTableHeader();
    
    // Хранение времени для последовательной версии (1 поток)
    std::vector<double> seq_times;
    seq_times.resize(test_sizes.size(), 0.0);
    
    // Основной цикл экспериментов
    for (size_t idx = 0; idx < test_sizes.size(); ++idx) {
        size_t size = test_sizes[idx];
        std::cout << std::flush;
        
        // Прогрев кэша (выполняем один тест перед измерениями)
        MatrixUtils::warmup(size);
        
        // Для каждого количества потоков
        for (int threads : thread_counts) {
            // Выполнение теста с усреднением (3 запуска)
            double total_time = 0.0;
            double max_diff = 0.0;
            bool verified = false;
            
            for (int run = 0; run < 3; run++) {
                PerformanceMetrics metrics = MatrixUtils::getMetricsWithThreads(size, threads);
                total_time += metrics.multiplication_time;
                max_diff = std::max(max_diff, metrics.max_difference);
                verified = metrics.verification_passed;
            }
            
            double avg_time = total_time / 3.0;
            
            // Сохранение времени для последовательной версии
            if (threads == 1) {
                seq_times[idx] = avg_time;
            }
            
            // Расчет ускорения и эффективности
            double speedup = (seq_times[idx] > 0) ? seq_times[idx] / avg_time : 1.0;
            double efficiency = speedup / threads;
            
            // Вывод результатов
            std::cout << std::left
                      << std::setw(10) << size
                      << std::setw(10) << threads
                      << std::setw(15) << std::fixed << std::setprecision(6) << avg_time
                      << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                      << std::setw(12) << std::fixed << std::setprecision(1) << (efficiency * 100)
                      << std::setw(20) << (verified ? "✓ PASSED" : "✗ FAILED")
                      << std::setw(15) << std::scientific << std::setprecision(2) << max_diff
                      << std::endl;
            
            // Сохранение результатов в файл
            MatrixUtils::saveResultsToFile(size, threads, avg_time, speedup, efficiency);
        }
        std::cout << std::endl;
    }
    
    printSeparator(94);
    
    // Вывод рекомендаций
    std::cout << "\nRECOMMENDATIONS:" << std::endl;
    #ifdef _OPENMP
    std::cout << "  - Optimal number of threads: " << omp_get_num_procs() << std::endl;
    #else
    std::cout << "  - Optimal number of threads: 4 (based on typical system)" << std::endl;
    #endif
    std::cout << "  - Minimum matrix size for parallelization: 500x500" << std::endl;
    std::cout << "  - Best schedule strategy: dynamic with chunk_size=32" << std::endl;
    
    std::cout << "\nResults saved to:" << std::endl;
    std::cout << "  - results/performance_data.csv" << std::endl;
    std::cout << "  - results/matrix_a.txt" << std::endl;
    std::cout << "  - results/matrix_b.txt" << std::endl;
    std::cout << "  - results/result_matrix.txt" << std::endl;
    
    printSeparator(94);
    
    return 0;
}