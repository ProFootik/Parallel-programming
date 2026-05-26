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
        while (ss >> val) row.push_back(val);
        if (!row.empty()) mat.push_back(row);
    }
    return mat;
}

bool is_square(const vector<vector<double>>& m) {
    if (m.empty()) return false;
    size_t n = m.size();
    return all_of(m.begin(), m.end(), [n](const auto& r) { return r.size() == n; });
}

void write_matrix(const string& filename, const vector<vector<double>>& mat) {
    ofstream file(filename);
    for (const auto& row : mat) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << fixed << setprecision(6) << row[i];
            if (i < row.size() - 1) file << " ";
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
            cerr << "Использование: mpirun -np <num> " << argv[0] << " <matrixA> <matrixB> <result>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<vector<double>> full_A, full_B;
    int n = 0;

    if (rank == 0) {
        full_A = read_matrix(argv[1]);
        full_B = read_matrix(argv[2]);

        if (!is_square(full_A) || !is_square(full_B) || full_A.size() != full_B.size()) {
            cerr << "Ошибка: матрицы не квадратные или разных размеров!" << endl;
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
    MPI_Bcast(B_flat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<vector<double>> B(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = B_flat[i * n + j];

    int rows_per_proc_base = n / world_size;
    int remainder = n % world_size;
    int local_rows = rows_per_proc_base + (rank < remainder ? 1 : 0);

    vector<int> sendcounts(world_size);
    vector<int> displs(world_size, 0);
    int current_displ = 0;
    for (int i = 0; i < world_size; ++i) {
        int proc_rows = rows_per_proc_base + (i < remainder ? 1 : 0);
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
    MPI_Scatterv(A_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        A_local_flat.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<vector<double>> A_local(local_rows, vector<double>(n));
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            A_local[i][j] = A_local_flat[i * n + j];

    vector<vector<double>> C_local(local_rows, vector<double>(n, 0.0));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = 0.0;
    if (rank == 0) start_time = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C_local[i][j] += A_local[i][k] * B[k][j];

    vector<double> C_local_flat(local_rows * n, 0.0);
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            C_local_flat[i * n + j] = C_local[i][j];

    vector<int> recvcounts = sendcounts;
    vector<int> recvdispls = displs;
    vector<double> C_flat(n * n, 0.0);

    MPI_Gatherv(C_local_flat.data(), local_rows * n, MPI_DOUBLE,
        C_flat.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double duration = MPI_Wtime() - start_time;

        vector<vector<double>> C(n, vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] = C_flat[i * n + j];

        write_matrix(argv[3], C);

        long long volume = (long long)n * n * n;
        cout << "Время выполнения: " << fixed << setprecision(4) << duration << " секунд" << endl;
        cout << "Количество процессов: " << world_size << endl;
        cout << "Размер матрицы: " << n << " x " << n << endl;
        cout << "Объем задачи: " << volume << " операций" << endl;
    }

    MPI_Finalize();
    return 0;
}