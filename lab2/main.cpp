#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>

#ifdef USE_BLAS
#include <openblas/cblas.h>
#endif

using cd = std::complex<double>;
using Clock = std::chrono::high_resolution_clock;

constexpr int N = 1024;      // размер матриц
constexpr int BS = 64;       // размер блока для оптимизации

// Генерация матрицы
void generate_matrix(std::vector<cd>& A) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto& x : A) {
        x = cd(dist(gen), dist(gen));
    }
}

// Наивное умножение
void matmul_naive(const std::vector<cd>& A,
                  const std::vector<cd>& B,
                  std::vector<cd>& C) {

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
void matmul_blas(const std::vector<cd>& A,
                 const std::vector<cd>& B,
                 std::vector<cd>& C) {

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
void matmul_blocked_parallel(const std::vector<cd>& A,
                    const std::vector<cd>& B,
                    std::vector<cd>& C) {

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
double measure_time(void (*func)(const std::vector<cd>&,
                                const std::vector<cd>&,
                                std::vector<cd>&),
                    const std::vector<cd>& A,
                    const std::vector<cd>& B,
                    std::vector<cd>& C) {

    auto start = Clock::now();
    func(A, B, C);
    auto end = Clock::now();

    return std::chrono::duration<double>(end - start).count();
}

int main() {

    std::vector<cd> A(N * N), B(N * N), C(N * N);

    generate_matrix(A);
    generate_matrix(B);

    double operations = 2.0 * N * N * N;

    while (true) {

        std::cout << "\nChoose multiplication method:\n";
        std::cout << "1 - Naive\n";
#ifdef USE_BLAS
        std::cout << "2 - BLAS\n";
#endif
        std::cout << "3 - Blocked OMP\n";
        std::cout << "0 - Exit\n";
        std::cout << "Choice: ";

        int choice;
        std::cin >> choice;

        if (choice == 0)
            break;

        std::fill(C.begin(), C.end(), 0.0);

        double t = 0;

        switch (choice) {
        case 1:
            t = measure_time(matmul_naive, A, B, C);
            std::cout << "Naive: " << operations / t * 1e-6 << " MFlops\n";
            break;

#ifdef USE_BLAS
        case 2:
            t = measure_time(matmul_blas, A, B, C);
            std::cout << "BLAS: " << operations / t * 1e-6 << " MFlops\n";
            break;
#endif

        case 3:
            t = measure_time(matmul_blocked_parallel, A, B, C);
            std::cout << "Blocked OMP: " << operations / t * 1e-6 << " MFlops\n";
            break;

        default:
            std::cout << "Invalid choice\n";
        }
    }

    return 0;
}