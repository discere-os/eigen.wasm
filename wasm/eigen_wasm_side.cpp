/*
 * eigen_wasm_side.cpp - SIDE_MODULE implementation for Eigen WASM
 *
 * Copyright (c) 2008-2024 Eigen Authors
 * Copyright (c) 2025 Superstruct Ltd, New Zealand
 * Licensed under Mozilla Public License v2.0
 *
 * SIMD-optimized linear algebra operations for WebAssembly
 */

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <emscripten.h>
#include <emscripten/html5_webgpu.h>
#include <wasm_simd128.h>
#include <cmath>
#include <memory>
#include <vector>

using namespace Eigen;

// Global state for performance optimization
static bool simd_enabled = true;
static bool webgpu_available = false;

// WebGPU context management
static void* webgpu_context = nullptr;

extern "C" {

// Feature detection
EMSCRIPTEN_KEEPALIVE
bool webgpu_available_check() {
#ifdef __EMSCRIPTEN_PTHREADS__
    return emscripten_webgpu_get_device() != 0;
#else
    // Check for WebGPU availability in single-threaded mode
    return false; // Will be overridden by external context
#endif
}

EMSCRIPTEN_KEEPALIVE
bool enable_simd(bool enable) {
    bool old_state = simd_enabled;
    simd_enabled = enable;
    return old_state;
}

// SIMD-optimized matrix multiplication
EMSCRIPTEN_KEEPALIVE
int matrix_multiply(double* a, double* b, double* result, int rows_a, int cols_a, int cols_b) {
    try {
        Map<MatrixXd> mat_a(a, rows_a, cols_a);
        Map<MatrixXd> mat_b(b, cols_a, cols_b);
        Map<MatrixXd> mat_result(result, rows_a, cols_b);

        // Eigen's optimized multiplication with SIMD when available
        mat_result.noalias() = mat_a * mat_b;

        return 0; // Success
    } catch (...) {
        return -1; // Error
    }
}

// Explicit SIMD matrix multiplication for benchmarking
EMSCRIPTEN_KEEPALIVE
int matrix_multiply_simd(float* a, float* b, float* result, int rows_a, int cols_a, int cols_b) {
    if (!simd_enabled) {
        return matrix_multiply((double*)a, (double*)b, (double*)result, rows_a, cols_a, cols_b);
    }

    try {
        // Use Eigen's vectorized operations which map to WASM SIMD
        Map<MatrixXf> mat_a(a, rows_a, cols_a);
        Map<MatrixXf> mat_b(b, cols_a, cols_b);
        Map<MatrixXf> mat_result(result, rows_a, cols_b);

        // Force vectorized computation
        mat_result.noalias() = mat_a * mat_b;

        return 0;
    } catch (...) {
        return -1;
    }
}

// SIMD-optimized vector dot product
EMSCRIPTEN_KEEPALIVE
double vector_dot(double* a, double* b, int size) {
    try {
        Map<VectorXd> vec_a(a, size);
        Map<VectorXd> vec_b(b, size);

        return vec_a.dot(vec_b);
    } catch (...) {
        return NAN;
    }
}

// Explicit SIMD dot product
EMSCRIPTEN_KEEPALIVE
float vector_dot_simd(float* a, float* b, int size) {
    if (!simd_enabled || size < 16) {
        // Fallback for small vectors
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    try {
        Map<VectorXf> vec_a(a, size);
        Map<VectorXf> vec_b(b, size);

        // Eigen will use SIMD when available
        return vec_a.dot(vec_b);
    } catch (...) {
        return NAN;
    }
}

// Single precision non-SIMD dot product
EMSCRIPTEN_KEEPALIVE
float vector_dot_single(float* a, float* b, int size) {
    try {
        Map<VectorXf> vec_a(a, size);
        Map<VectorXf> vec_b(b, size);
        // Non-SIMD path - should give same result as SIMD
        return vec_a.dot(vec_b);
    } catch (...) {
        return NAN;
    }
}

// Matrix inversion using LU decomposition
EMSCRIPTEN_KEEPALIVE
int matrix_invert(double* matrix, double* result, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);
        Map<MatrixXd> inv(result, size, size);

        // Use Eigen's optimized LU decomposition
        FullPivLU<MatrixXd> lu(mat);
        if (!lu.isInvertible()) {
            return -2; // Matrix not invertible
        }

        inv = lu.inverse();
        return 0;
    } catch (...) {
        return -1;
    }
}

// Matrix decomposition (LU, QR, SVD)
EMSCRIPTEN_KEEPALIVE
int matrix_decompose(double* matrix, int rows, int cols, int decomp_type,
                    double* result_l, double* result_u, double* result_p) {
    try {
        Map<MatrixXd> mat(matrix, rows, cols);

        switch (decomp_type) {
            case 0: { // LU decomposition
                PartialPivLU<MatrixXd> lu(mat);
                Map<MatrixXd> L(result_l, rows, std::min(rows, cols));
                Map<MatrixXd> U(result_u, std::min(rows, cols), cols);

                MatrixXd P_L_U = lu.matrixLU();
                L = MatrixXd::Identity(rows, std::min(rows, cols));
                L += P_L_U.triangularView<StrictlyLower>();
                U = P_L_U.triangularView<Upper>();

                return 0;
            }
            case 1: { // QR decomposition
                HouseholderQR<MatrixXd> qr(mat);
                Map<MatrixXd> Q(result_l, rows, rows);
                Map<MatrixXd> R(result_u, rows, cols);

                Q = qr.householderQ();
                R = qr.matrixQR().triangularView<Upper>();

                return 0;
            }
            case 2: { // SVD
                JacobiSVD<MatrixXd> svd(mat, ComputeFullU | ComputeFullV);
                Map<MatrixXd> U(result_l, rows, rows);
                Map<MatrixXd> V(result_u, cols, cols);
                Map<VectorXd> S(result_p, std::min(rows, cols));

                U = svd.matrixU();
                V = svd.matrixV();
                S = svd.singularValues();

                return 0;
            }
            default:
                return -2; // Invalid decomposition type
        }
    } catch (...) {
        return -1;
    }
}

// Eigenvalue computation
EMSCRIPTEN_KEEPALIVE
int eigenvalues(double* matrix, double* eigenvals, double* eigenvecs, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);

        EigenSolver<MatrixXd> solver(mat);
        if (solver.info() != Success) {
            return -2; // Convergence failed
        }

        // Real parts of eigenvalues
        Map<VectorXd> vals(eigenvals, size);
        vals = solver.eigenvalues().real();

        // Real parts of eigenvectors (if requested)
        if (eigenvecs != nullptr) {
            Map<MatrixXd> vecs(eigenvecs, size, size);
            vecs = solver.eigenvectors().real();
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

// Linear system solver
EMSCRIPTEN_KEEPALIVE
int matrix_solve(double* A, double* b, double* x, int size) {
    try {
        Map<MatrixXd> mat_A(A, size, size);
        Map<VectorXd> vec_b(b, size);
        Map<VectorXd> vec_x(x, size);

        // Use Eigen's optimized solver with partial pivoting
        PartialPivLU<MatrixXd> solver(mat_A);
        vec_x = solver.solve(vec_b);

        // Check solution accuracy
        double relative_error = (mat_A * vec_x - vec_b).norm() / vec_b.norm();
        if (relative_error > 1e-10) {
            return -2; // Poor solution quality
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

// Matrix transpose (in-place when possible)
EMSCRIPTEN_KEEPALIVE
int matrix_transpose(double* matrix, double* result, int rows, int cols) {
    try {
        Map<MatrixXd> mat(matrix, rows, cols);
        Map<MatrixXd> trans(result, cols, rows);

        trans = mat.transpose();
        return 0;
    } catch (...) {
        return -1;
    }
}

// Matrix determinant
EMSCRIPTEN_KEEPALIVE
double matrix_determinant(double* matrix, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);
        return mat.determinant();
    } catch (...) {
        return NAN;
    }
}

// Performance monitoring
EMSCRIPTEN_KEEPALIVE
int memory_stats(int* allocated_mb, int* peak_mb) {
    // Simplified memory tracking
    *allocated_mb = 0; // Would need custom allocator for accurate tracking
    *peak_mb = 0;
    return 0;
}

} // extern "C"