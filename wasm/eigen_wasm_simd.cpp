/*
 * eigen_wasm_simd.cpp - Enhanced SIMD optimizations for Eigen WASM
 *
 * Copyright (c) 2008-2024 Eigen Authors
 * Copyright (c) 2025 Superstruct Ltd, New Zealand
 * Licensed under Mozilla Public License v2.0
 *
 * High-performance SIMD implementations targeting 4-8x speedups
 */

#include <Eigen/Dense>
#include <emscripten.h>
#include <wasm_simd128.h>
#include <cmath>
#include <memory>
#include <cstring>

using namespace Eigen;

// SIMD-optimized memory operations
extern "C" {

// Feature detection for SIMD capabilities
EMSCRIPTEN_KEEPALIVE
bool simd_supported() {
#ifdef __wasm_simd128__
    return true;
#else
    return false;
#endif
}

// SIMD-optimized vector operations
EMSCRIPTEN_KEEPALIVE
void simd_vector_add_f32(const float* a, const float* b, float* result, int size) {
#ifdef __wasm_simd128__
    int simd_end = (size / 4) * 4;

    // Process 4 floats at a time
    for (int i = 0; i < simd_end; i += 4) {
        v128_t va = wasm_v128_load(&a[i]);
        v128_t vb = wasm_v128_load(&b[i]);
        v128_t vr = wasm_f32x4_add(va, vb);
        wasm_v128_store(&result[i], vr);
    }

    // Handle remainder
    for (int i = simd_end; i < size; i++) {
        result[i] = a[i] + b[i];
    }
#else
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
#endif
}

// SIMD-optimized dot product with better memory safety
EMSCRIPTEN_KEEPALIVE
float simd_dot_product_f32(const float* a, const float* b, int size) {
    if (size <= 0 || !a || !b) return 0.0f;

#ifdef __wasm_simd128__
    v128_t sum_vec = wasm_f32x4_splat(0.0f);
    int simd_end = (size / 4) * 4;

    // Process 4 floats at a time
    for (int i = 0; i < simd_end; i += 4) {
        v128_t va = wasm_v128_load(&a[i]);
        v128_t vb = wasm_v128_load(&b[i]);
        v128_t prod = wasm_f32x4_mul(va, vb);
        sum_vec = wasm_f32x4_add(sum_vec, prod);
    }

    // Extract horizontal sum
    float temp[4];
    wasm_v128_store(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Handle remainder
    for (int i = simd_end; i < size; i++) {
        sum += a[i] * b[i];
    }

    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// SIMD-optimized matrix multiplication (small blocks)
EMSCRIPTEN_KEEPALIVE
int simd_matrix_multiply_f32(const float* a, const float* b, float* c,
                            int m, int k, int n) {
    if (!a || !b || !c || m <= 0 || k <= 0 || n <= 0) return -1;

    try {
        // Use Eigen with explicit vectorization hints
        Map<const MatrixXf> mat_a(a, m, k);
        Map<const MatrixXf> mat_b(b, k, n);
        Map<MatrixXf> mat_c(c, m, n);

        // Eigen's optimized multiplication with SIMD
        mat_c.noalias() = mat_a * mat_b;

        return 0;
    } catch (...) {
        return -1;
    }
}

// Memory-safe matrix transpose with SIMD optimization
EMSCRIPTEN_KEEPALIVE
int simd_matrix_transpose_f32(const float* input, float* output, int rows, int cols) {
    if (!input || !output || rows <= 0 || cols <= 0) return -1;

    try {
        // Block-wise transpose for cache efficiency
        const int BLOCK_SIZE = 8;

        for (int i = 0; i < rows; i += BLOCK_SIZE) {
            for (int j = 0; j < cols; j += BLOCK_SIZE) {
                int max_i = std::min(i + BLOCK_SIZE, rows);
                int max_j = std::min(j + BLOCK_SIZE, cols);

                // Transpose within block
                for (int ii = i; ii < max_i; ii++) {
                    for (int jj = j; jj < max_j; jj++) {
                        output[jj * rows + ii] = input[ii * cols + jj];
                    }
                }
            }
        }

        return 0;
    } catch (...) {
        return -1;
    }
}

// High-performance matrix-vector multiplication
EMSCRIPTEN_KEEPALIVE
int simd_matrix_vector_multiply_f32(const float* matrix, const float* vector,
                                   float* result, int rows, int cols) {
    if (!matrix || !vector || !result || rows <= 0 || cols <= 0) return -1;

    try {
        Map<const MatrixXf> mat(matrix, rows, cols);
        Map<const VectorXf> vec(vector, cols);
        Map<VectorXf> res(result, rows);

        res.noalias() = mat * vec;

        return 0;
    } catch (...) {
        return -1;
    }
}

// SIMD-optimized element-wise operations
EMSCRIPTEN_KEEPALIVE
void simd_vector_scale_f32(const float* input, float scale, float* output, int size) {
    if (!input || !output || size <= 0) return;

#ifdef __wasm_simd128__
    v128_t scale_vec = wasm_f32x4_splat(scale);
    int simd_end = (size / 4) * 4;

    for (int i = 0; i < simd_end; i += 4) {
        v128_t data = wasm_v128_load(&input[i]);
        v128_t result = wasm_f32x4_mul(data, scale_vec);
        wasm_v128_store(&output[i], result);
    }

    // Handle remainder
    for (int i = simd_end; i < size; i++) {
        output[i] = input[i] * scale;
    }
#else
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * scale;
    }
#endif
}

// Parallel reduction sum with SIMD
EMSCRIPTEN_KEEPALIVE
float simd_vector_sum_f32(const float* data, int size) {
    if (!data || size <= 0) return 0.0f;

#ifdef __wasm_simd128__
    v128_t sum_vec = wasm_f32x4_splat(0.0f);
    int simd_end = (size / 4) * 4;

    // SIMD accumulation
    for (int i = 0; i < simd_end; i += 4) {
        v128_t chunk = wasm_v128_load(&data[i]);
        sum_vec = wasm_f32x4_add(sum_vec, chunk);
    }

    // Horizontal sum
    float temp[4];
    wasm_v128_store(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Remainder
    for (int i = simd_end; i < size; i++) {
        sum += data[i];
    }

    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
#endif
}

// Fast memory copy with SIMD alignment
EMSCRIPTEN_KEEPALIVE
void simd_memcpy_aligned(const void* src, void* dst, size_t bytes) {
    if (!src || !dst || bytes == 0) return;

#ifdef __wasm_simd128__
    const uint8_t* src_ptr = (const uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;

    // Process 16-byte chunks
    size_t simd_bytes = (bytes / 16) * 16;
    for (size_t i = 0; i < simd_bytes; i += 16) {
        v128_t chunk = wasm_v128_load(&src_ptr[i]);
        wasm_v128_store(&dst_ptr[i], chunk);
    }

    // Handle remainder
    for (size_t i = simd_bytes; i < bytes; i++) {
        dst_ptr[i] = src_ptr[i];
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

// SIMD matrix norm calculation
EMSCRIPTEN_KEEPALIVE
float simd_matrix_frobenius_norm_f32(const float* matrix, int rows, int cols) {
    if (!matrix || rows <= 0 || cols <= 0) return 0.0f;

    int size = rows * cols;

#ifdef __wasm_simd128__
    v128_t sum_vec = wasm_f32x4_splat(0.0f);
    int simd_end = (size / 4) * 4;

    for (int i = 0; i < simd_end; i += 4) {
        v128_t chunk = wasm_v128_load(&matrix[i]);
        v128_t squared = wasm_f32x4_mul(chunk, chunk);
        sum_vec = wasm_f32x4_add(sum_vec, squared);
    }

    // Horizontal sum
    float temp[4];
    wasm_v128_store(temp, sum_vec);
    float sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Remainder
    for (int i = simd_end; i < size; i++) {
        sum += matrix[i] * matrix[i];
    }

    return std::sqrt(sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += matrix[i] * matrix[i];
    }
    return std::sqrt(sum);
#endif
}

// Benchmark functions for performance testing
EMSCRIPTEN_KEEPALIVE
double benchmark_simd_dot_product(int size, int iterations) {
    if (size <= 0 || iterations <= 0) return -1.0;

    std::unique_ptr<float[]> a(new float[size]);
    std::unique_ptr<float[]> b(new float[size]);

    // Initialize with random data
    for (int i = 0; i < size; i++) {
        a[i] = (float)(rand()) / RAND_MAX;
        b[i] = (float)(rand()) / RAND_MAX;
    }

    // Warm up
    volatile float result = simd_dot_product_f32(a.get(), b.get(), size);
    (void)result; // Suppress unused variable warning

    // Benchmark
    double start_time = emscripten_get_now();

    for (int i = 0; i < iterations; i++) {
        result = simd_dot_product_f32(a.get(), b.get(), size);
    }

    double end_time = emscripten_get_now();

    return end_time - start_time;
}

EMSCRIPTEN_KEEPALIVE
double benchmark_simd_matrix_multiply(int size, int iterations) {
    if (size <= 0 || iterations <= 0) return -1.0;

    int matrix_size = size * size;
    std::unique_ptr<float[]> a(new float[matrix_size]);
    std::unique_ptr<float[]> b(new float[matrix_size]);
    std::unique_ptr<float[]> c(new float[matrix_size]);

    // Initialize
    for (int i = 0; i < matrix_size; i++) {
        a[i] = (float)(rand()) / RAND_MAX;
        b[i] = (float)(rand()) / RAND_MAX;
    }

    // Warm up
    int result = simd_matrix_multiply_f32(a.get(), b.get(), c.get(), size, size, size);
    if (result != 0) return -1.0;

    // Benchmark
    double start_time = emscripten_get_now();

    for (int i = 0; i < iterations; i++) {
        result = simd_matrix_multiply_f32(a.get(), b.get(), c.get(), size, size, size);
        if (result != 0) return -1.0;
    }

    double end_time = emscripten_get_now();

    return end_time - start_time;
}

} // extern "C"