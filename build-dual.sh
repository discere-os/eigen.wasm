#!/bin/bash
# build-dual.sh - Dual build system for eigen.wasm
#
# Copyright (c) 2008-2024 Eigen Authors
# Copyright (c) 2025 Superstruct Ltd, New Zealand
# Licensed under Mozilla Public License v2.0

set -euo pipefail

VARIANT="${1:-all}"
BUILD_DIR="${BUILD_DIR:-./wasm-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-./install}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking build prerequisites for Eigen..."

    if ! command -v emcc &> /dev/null; then
        log_error "Emscripten not found. Please install and activate EMSDK."
        exit 1
    fi

    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake."
        exit 1
    fi

    # Check Emscripten version
    local emcc_version=$(emcc --version | head -n1 | grep -oP '\d+\.\d+\.\d+')
    log_info "Using Emscripten ${emcc_version}"

    log_success "Prerequisites check completed"
}

# Build SIDE_MODULE (production) - just the essential linear algebra functions
build_side_module() {
    log_info "Building eigen-side.wasm for production..."
    mkdir -p "${BUILD_DIR}-side"
    cd "${BUILD_DIR}-side"

    # Configure CMake for SIDE_MODULE with SIMD and WebGPU
    emcmake cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SIDE_MODULE=ON \
        -DEIGEN_BUILD_DOC=OFF \
        -DEIGEN_BUILD_PKGCONFIG=OFF \
        -DEIGEN_BUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX="${PWD}/../install" \
        -DCMAKE_C_FLAGS="-O3 -flto -msimd128 -fPIC" \
        -DCMAKE_CXX_FLAGS="-O3 -flto -msimd128 -fPIC -DEIGEN_NO_IO -DEIGEN_DONT_VECTORIZE=0" \
        -DCMAKE_EXE_LINKER_FLAGS="-sSIDE_MODULE=1 -sSTANDALONE_WASM=1 -sUSE_WEBGPU=1"

    # Build our WASM wrapper instead of full library
    emcc ../wasm/eigen_wasm_side.cpp \
        -I.. -I../Eigen \
        -O3 -flto -msimd128 -fPIC \
        -DEIGEN_NO_IO \
        -DEIGEN_DONT_VECTORIZE=0 \
        -DEIGEN_USE_WEBGPU \
        -sSIDE_MODULE=1 \
        -sSTANDALONE_WASM=1 \
        -sUSE_WEBGPU=1 \
        -sEXPORTED_FUNCTIONS='["_matrix_multiply","_matrix_multiply_simd","_vector_dot","_vector_dot_simd","_vector_dot_single","_matrix_invert","_matrix_decompose","_eigenvalues","_matrix_solve","_matrix_transpose","_matrix_determinant","_webgpu_available_check","_enable_simd"]' \
        -o eigen-side.wasm

    # Install artifacts
    mkdir -p "${INSTALL_PREFIX}/wasm"
    cp eigen-side.wasm "${INSTALL_PREFIX}/wasm/"

    log_success "SIDE_MODULE: ${INSTALL_PREFIX}/wasm/eigen-side.wasm ($(du -h ${INSTALL_PREFIX}/wasm/eigen-side.wasm | cut -f1))"
    cd ..
}

# Build MAIN_MODULE (testing/NPM) - full featured version
build_main_module() {
    log_info "Building eigen-main.js for testing..."
    mkdir -p "${BUILD_DIR}-main"
    cd "${BUILD_DIR}-main"

    # Configure CMake for MAIN_MODULE
    emcmake cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_MAIN_MODULE=ON \
        -DEIGEN_BUILD_DOC=OFF \
        -DEIGEN_BUILD_PKGCONFIG=OFF \
        -DEIGEN_BUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX="${PWD}/../install" \
        -DCMAKE_C_FLAGS="-O3 -flto -msimd128" \
        -DCMAKE_CXX_FLAGS="-O3 -flto -msimd128 -DEIGEN_DONT_VECTORIZE=0" \
        -DCMAKE_EXE_LINKER_FLAGS="-sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=EigenModule -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sMAXIMUM_MEMORY=536870912 -sUSE_WEBGPU=1 -sASYNCIFY=1"

    # Build comprehensive MAIN_MODULE wrapper
    emcc ../wasm/eigen_wasm_main.cpp \
        -I.. -I../Eigen \
        -O3 -flto -msimd128 \
        -DEIGEN_DONT_VECTORIZE=0 \
        -DEIGEN_USE_WEBGPU \
        -sMODULARIZE=1 \
        -sEXPORT_ES6=1 \
        -sEXPORT_NAME="EigenModule" \
        -sEXPORTED_FUNCTIONS='["_matrix_multiply","_matrix_multiply_simd","_vector_dot","_vector_dot_simd","_vector_dot_single","_matrix_invert","_matrix_decompose","_eigenvalues","_matrix_solve","_matrix_transpose","_matrix_determinant","_webgpu_available_check","_enable_simd","_malloc","_free","_memory_stats"]' \
        -sEXPORTED_RUNTIME_METHODS='["cwrap","ccall","UTF8ToString","HEAPU8","HEAPF32","HEAPF64"]' \
        -sALLOW_MEMORY_GROWTH=1 \
        -sINITIAL_MEMORY=67108864 \
        -sMAXIMUM_MEMORY=536870912 \
        -sUSE_WEBGPU=1 \
        -sASYNCIFY=1 \
        -o eigen-main.js

    # Install artifacts
    mkdir -p "${INSTALL_PREFIX}/wasm"
    cp eigen-main.js "${INSTALL_PREFIX}/wasm/"
    cp eigen-main.wasm "${INSTALL_PREFIX}/wasm/"

    log_success "MAIN_MODULE: ${INSTALL_PREFIX}/wasm/eigen-main.js (JS: $(du -h ${INSTALL_PREFIX}/wasm/eigen-main.js | cut -f1), WASM: $(du -h ${INSTALL_PREFIX}/wasm/eigen-main.wasm | cut -f1))"
    cd ..
}

# Performance analysis
analyze_performance() {
    log_info "Analyzing build performance..."

    if [ -f "${INSTALL_PREFIX}/wasm/eigen-side.wasm" ]; then
        local side_size=$(stat -c%s "${INSTALL_PREFIX}/wasm/eigen-side.wasm")
        log_info "SIDE_MODULE size: $(numfmt --to=iec $side_size) (optimized for dynamic loading)"
    fi

    if [ -f "${INSTALL_PREFIX}/wasm/eigen-main.wasm" ]; then
        local main_size=$(stat -c%s "${INSTALL_PREFIX}/wasm/eigen-main.wasm")
        log_info "MAIN_MODULE size: $(numfmt --to=iec $main_size) (includes runtime)"
    fi
}

case "$VARIANT" in
    side)
        check_prerequisites && build_side_module && analyze_performance
        ;;
    main)
        check_prerequisites && build_main_module && analyze_performance
        ;;
    all)
        check_prerequisites && build_side_module && build_main_module && analyze_performance
        ;;
    clean)
        rm -rf "${BUILD_DIR}"* "${INSTALL_PREFIX}" wasm-build*
        log_success "Build artifacts cleaned"
        ;;
    *)
        echo "Usage: $0 [side|main|all|clean]"
        echo "  side  - Build production SIDE_MODULE"
        echo "  main  - Build testing MAIN_MODULE"
        echo "  all   - Build both modules"
        echo "  clean - Remove all build artifacts"
        exit 1
        ;;
esac