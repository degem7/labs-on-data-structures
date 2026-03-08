#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <windows.h>

#ifdef USE_BLAS
#include <openblas/cblas.h>
#endif

using namespace std;
using cd = std::complex<double>;
using Clock = std::chrono::high_resolution_clock;

constexpr int N = 1024;      // размер матриц
constexpr int BS = 64;       // размер блока для оптимизации

// Генерация матрицы
void generate_matrix(vector<cd>& A) {
    mt19937 gen(42);
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto& x : A) {
        x = cd(dist(gen), dist(gen));
    }
}

// Наивное умножение
void matmul_naive(const vector<cd>& A,
                  const vector<cd>& B,
                  vector<cd>& C) {

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cd sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// BLAS
#ifdef USE_BLAS
void matmul_blas(const vector<cd>& A,
                 const vector<cd>& B,
                 vector<cd>& C) {

    cd alpha = 1.0;
    cd beta  = 0.0;

    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        &alpha,
        A.data(), N,
        B.data(), N,
        &beta,
        C.data(), N
    );
}
#endif

// Блочное умножение с порядком i-k-j
void matmul_blocked_parallel(const vector<cd>& A,
                    const vector<cd>& B,
                    vector<cd>& C) {

#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {

                for (int i = ii; i < ii + BS; ++i) {
                    for (int k = kk; k < kk + BS; ++k) {
                        cd a = A[i * N + k];
                        for (int j = jj; j < jj + BS; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }

            }
        }
    }
}

// Замер времени
double measure_time(void (*func)(const vector<cd>&,
                                const vector<cd>&,
                                vector<cd>&),
                    const vector<cd>& A,
                    const vector<cd>& B,
                    vector<cd>& C) {

    auto start = Clock::now();
    func(A, B, C);
    auto end = Clock::now();

    return chrono::duration<double>(end - start).count();
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    vector<cd> A(N * N), B(N * N), C(N * N);

    generate_matrix(A);
    generate_matrix(B);

    double operations = 2.0 * N * N * N;

    while (true) {

        cout << "\nВыберите метод умножения:\n";
        cout << "1 - Naive\n";
#ifdef USE_BLAS
        cout << "2 - BLAS\n";
#endif
        cout << "3 - Blocked OMP\n";
        cout << "0 - Выход\n";
        cout << "Выбор: ";

        int choice;
        cin >> choice;

        if (choice == 0)
            break;

        fill(C.begin(), C.end(), 0.0);

        double t = 0;

        switch (choice) {
        case 1:
            t = measure_time(matmul_naive, A, B, C);
            cout << "Naive: " << operations / t * 1e-6 << " MFlops\n";
            break;

#ifdef USE_BLAS
        case 2:
            t = measure_time(matmul_blas, A, B, C);
            cout << "BLAS: " << operations / t * 1e-6 << " MFlops\n";
            break;
#endif

        case 3:
            t = measure_time(matmul_blocked_parallel, A, B, C);
            cout << "Blocked OMP: " << operations / t * 1e-6 << " MFlops\n";
            break;

        default:
            cout << "Неправильный выбор\n";
        }
    }

    cout << "Бергер Денис Максимович, 090304-РПИа-025" << endl;
    while(getchar() != '\n');

    return 0;
}