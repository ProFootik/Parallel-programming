#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

vector<double> readMatrix(const string& filename, int size) {
    vector<double> matrix(size * size);
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        exit(1);
    }
    
    for (int i = 0; i < size * size; i++) {
        file >> matrix[i];
    }
    
    file.close();
    return matrix;
}

void writeMatrix(const string& filename, const vector<double>& matrix, int size) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Ошибка создания файла: " << filename << endl;
        exit(1);
    }
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << fixed << setprecision(6) << matrix[i * size + j];
            if (j != size - 1) file << " ";
        }
        file << endl;
    }
    
    file.close();
}

double multiplyCUDA(const vector<double>& h_A, const vector<double>& h_B, 
                    vector<double>& h_C, int n, int blockX, int blockY) {
    double *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(double);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(blockX, blockY);
    dim3 gridSize((n + blockX - 1) / blockX, (n + blockY - 1) / blockY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 1000.0; 
}

double computeGflops(int n, double timeSeconds) {
    double flops = 2.0 * n * n * n;  
    return flops / (timeSeconds * 1e9);
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    vector<pair<int,int>> blockConfigs = {
        {8, 8},    
        {16, 16},  
        {32, 32}   
    };
    
    system("mkdir -p results_cuda");

    ofstream statsFile("results_cuda/performance_results.txt");
    statsFile << "# Результаты умножения матриц с использованием CUDA\n";
    statsFile << "# Размер\tКонфигурация\tВремя(сек)\tGFLOPS\n";
    
    cout << "\n" << string(70, '=') << endl;
    cout << "УМНОЖЕНИЕ МАТРИЦ С ИСПОЛЬЗОВАНИЕМ CUDA" << endl;
    cout << string(70, '=') << endl;
    cout << left << setw(12) << "Размер" 
         << setw(18) << "Конфигурация" 
         << setw(16) << "Время (сек)" 
         << setw(16) << "GFLOPS" << endl;
    cout << string(62, '-') << endl;

    for (int size : sizes) {
        cout << "\nОбработка матриц размера " << size << "x" << size << ":" << endl;

        string fileA = "data/matrices/matrix_A_" + to_string(size) + ".txt";
        string fileB = "data/matrices/matrix_B_" + to_string(size) + ".txt";
        
        vector<double> A = readMatrix(fileA, size);
        vector<double> B = readMatrix(fileB, size);
        vector<double> C(size * size);

        for (auto& config : blockConfigs) {
            int blockX = config.first;
            int blockY = config.second;

            double timeSec = multiplyCUDA(A, B, C, size, blockX, blockY);
            double gflops = computeGflops(size, timeSec);

            string configStr = to_string(blockX) + "x" + to_string(blockY);

            cout << left << setw(12) << size 
                 << setw(18) << configStr 
                 << setw(16) << fixed << setprecision(6) << timeSec 
                 << setw(16) << fixed << setprecision(2) << gflops << endl;
            
            // Сохранение результата умножения
            string resultFile = "results_cuda/result_" + to_string(size) + "_" + configStr + ".txt";
            writeMatrix(resultFile, C, size);
            
            // Запись в файл статистики
            statsFile << size << "\t" 
                     << configStr << "\t" 
                     << timeSec << "\t" 
                     << gflops << endl;
        }
    }
    
    statsFile.close();
    
    cout << "\n" << string(70, '=') << endl;
    cout << " Все вычисления завершены!" << endl;
    cout << " Результаты сохранены в папку 'results_cuda/'" << endl;
    cout << " Статистика сохранена в 'results_cuda/performance_results.txt'" << endl;
    cout << string(70, '=') << endl;
    
    return 0;
}
