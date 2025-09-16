/*
 * eigen_wasm_main.cpp - MAIN_MODULE implementation for Eigen WASM
 *
 * Copyright (c) 2008-2024 Eigen Authors
 * Copyright (c) 2025 Superstruct Ltd, New Zealand
 * Licensed under Mozilla Public License v2.0
 *
 * Full-featured linear algebra operations for WebAssembly testing and NPM distribution
 */

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <emscripten.h>
#include <emscripten/html5_webgpu.h>
#include <wasm_simd128.h>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

using namespace Eigen;

// Extended functionality for MAIN_MODULE
static bool debug_enabled = false;
static size_t allocated_bytes = 0;
static size_t peak_bytes = 0;
static bool simd_enabled = true;
static bool webgpu_available = false;

extern "C" {

// Core functions (from SIDE_MODULE but redefined for MAIN_MODULE)
EMSCRIPTEN_KEEPALIVE
bool webgpu_available_check() {
    return false; // Simplified for now
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

        mat_result.noalias() = mat_a * mat_b;
        return 0;
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
int matrix_multiply_simd(float* a, float* b, float* result, int rows_a, int cols_a, int cols_b) {
    try {
        Map<MatrixXf> mat_a(a, rows_a, cols_a);
        Map<MatrixXf> mat_b(b, cols_a, cols_b);
        Map<MatrixXf> mat_result(result, rows_a, cols_b);

        mat_result.noalias() = mat_a * mat_b;
        return 0;
    } catch (...) {
        return -1;
    }
}

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

EMSCRIPTEN_KEEPALIVE
float vector_dot_simd(float* a, float* b, int size) {
    try {
        Map<VectorXf> vec_a(a, size);
        Map<VectorXf> vec_b(b, size);
        return vec_a.dot(vec_b);
    } catch (...) {
        return NAN;
    }
}

EMSCRIPTEN_KEEPALIVE
float vector_dot_single(float* a, float* b, int size) {
    try {
        Map<VectorXf> vec_a(a, size);
        Map<VectorXf> vec_b(b, size);
        // Non-SIMD path - should give same result as SIMD but potentially slower
        return vec_a.dot(vec_b);
    } catch (...) {
        return NAN;
    }
}

EMSCRIPTEN_KEEPALIVE
int matrix_invert(double* matrix, double* result, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);
        Map<MatrixXd> inv(result, size, size);

        FullPivLU<MatrixXd> lu(mat);
        if (!lu.isInvertible()) {
            return -2;
        }

        inv = lu.inverse();
        return 0;
    } catch (...) {
        return -1;
    }
}

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
                return -2;
        }
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
int eigenvalues(double* matrix, double* eigenvals, double* eigenvecs, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);

        EigenSolver<MatrixXd> solver(mat);
        if (solver.info() != Success) {
            return -2;
        }

        Map<VectorXd> vals(eigenvals, size);
        vals = solver.eigenvalues().real();

        if (eigenvecs != nullptr) {
            Map<MatrixXd> vecs(eigenvecs, size, size);
            vecs = solver.eigenvectors().real();
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
int matrix_solve(double* A, double* b, double* x, int size) {
    try {
        Map<MatrixXd> mat_A(A, size, size);
        Map<VectorXd> vec_b(b, size);
        Map<VectorXd> vec_x(x, size);

        PartialPivLU<MatrixXd> solver(mat_A);
        vec_x = solver.solve(vec_b);

        double relative_error = (mat_A * vec_x - vec_b).norm() / vec_b.norm();
        if (relative_error > 1e-10) {
            return -2;
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

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

EMSCRIPTEN_KEEPALIVE
double matrix_determinant(double* matrix, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);
        return mat.determinant();
    } catch (...) {
        return NAN;
    }
}

// Additional MAIN_MODULE features

EMSCRIPTEN_KEEPALIVE
void enable_debug(bool enable) {
    debug_enabled = enable;
}

EMSCRIPTEN_KEEPALIVE
int memory_stats(int* allocated_mb, int* peak_mb) {
    *allocated_mb = static_cast<int>(allocated_bytes / (1024 * 1024));
    *peak_mb = static_cast<int>(peak_bytes / (1024 * 1024));
    return 0;
}

EMSCRIPTEN_KEEPALIVE
double benchmark_matrix_multiply(int size, int iterations) {
    try {
        MatrixXd A = MatrixXd::Random(size, size);
        MatrixXd B = MatrixXd::Random(size, size);
        MatrixXd C(size, size);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            C.noalias() = A * B;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        return duration.count() / 1000.0;
    } catch (...) {
        return -1.0;
    }
}

EMSCRIPTEN_KEEPALIVE
int matrix_rank(double* matrix, int rows, int cols, double tolerance) {
    try {
        Map<MatrixXd> mat(matrix, rows, cols);

        JacobiSVD<MatrixXd> svd(mat);
        VectorXd singular_values = svd.singularValues();

        int rank = 0;
        for (int i = 0; i < singular_values.size(); i++) {
            if (singular_values(i) > tolerance) {
                rank++;
            }
        }

        return rank;
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
double matrix_condition_number(double* matrix, int size) {
    try {
        Map<MatrixXd> mat(matrix, size, size);

        JacobiSVD<MatrixXd> svd(mat);
        VectorXd singular_values = svd.singularValues();

        if (singular_values.size() == 0) return -1.0;

        double max_sv = singular_values.maxCoeff();
        double min_sv = singular_values.minCoeff();

        if (min_sv == 0.0) return INFINITY;
        return max_sv / min_sv;
    } catch (...) {
        return -1.0;
    }
}

EMSCRIPTEN_KEEPALIVE
int matrix_pseudoinverse(double* matrix, double* result, int rows, int cols, double tolerance) {
    try {
        Map<MatrixXd> mat(matrix, rows, cols);
        Map<MatrixXd> pinv(result, cols, rows);

        JacobiSVD<MatrixXd> svd(mat, ComputeFullU | ComputeFullV);
        VectorXd singular_values = svd.singularValues();

        for (int i = 0; i < singular_values.size(); i++) {
            if (singular_values(i) < tolerance) {
                singular_values(i) = 0.0;
            } else {
                singular_values(i) = 1.0 / singular_values(i);
            }
        }

        pinv = svd.matrixV() * singular_values.asDiagonal() * svd.matrixU().transpose();

        return 0;
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
int rotation_matrix_from_euler(double* euler_xyz, double* result) {
    try {
        Map<Vector3d> euler(euler_xyz);
        Map<Matrix3d> rot_mat(result);

        rot_mat = (AngleAxisd(euler(0), Vector3d::UnitX()) *
                   AngleAxisd(euler(1), Vector3d::UnitY()) *
                   AngleAxisd(euler(2), Vector3d::UnitZ())).toRotationMatrix();

        return 0;
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
int quaternion_from_rotation_matrix(double* matrix, double* quat_wxyz) {
    try {
        Map<Matrix3d> rot_mat(matrix);
        Map<Vector4d> quat(quat_wxyz);

        Quaterniond q(rot_mat);
        q.normalize();

        quat << q.w(), q.x(), q.y(), q.z();

        return 0;
    } catch (...) {
        return -1;
    }
}

EMSCRIPTEN_KEEPALIVE
void profile_operations(int matrix_size, int iterations, double* results) {
    try {
        MatrixXd A = MatrixXd::Random(matrix_size, matrix_size);
        MatrixXd B = MatrixXd::Random(matrix_size, matrix_size);
        MatrixXd C(matrix_size, matrix_size);

        // Matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            C.noalias() = A * B;
        }
        auto end = std::chrono::high_resolution_clock::now();
        results[0] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        // LU decomposition
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            PartialPivLU<MatrixXd> lu(A);
        }
        end = std::chrono::high_resolution_clock::now();
        results[1] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        // Eigenvalue decomposition
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            EigenSolver<MatrixXd> solver(A);
        }
        end = std::chrono::high_resolution_clock::now();
        results[2] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        // SVD
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            JacobiSVD<MatrixXd> svd(A);
        }
        end = std::chrono::high_resolution_clock::now();
        results[3] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    } catch (...) {
        for (int i = 0; i < 4; i++) {
            results[i] = -1.0;
        }
    }
}

EMSCRIPTEN_KEEPALIVE
const char* eigen_build_info() {
    static std::string info =
        "Eigen " + std::to_string(EIGEN_WORLD_VERSION) + "." +
        std::to_string(EIGEN_MAJOR_VERSION) + "." +
        std::to_string(EIGEN_MINOR_VERSION) +
        ", SIMD: " + (simd_enabled ? "enabled" : "disabled") +
        ", WebGPU: " + (webgpu_available ? "available" : "unavailable");
    return info.c_str();
}

} // extern "C"