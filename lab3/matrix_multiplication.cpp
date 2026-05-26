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