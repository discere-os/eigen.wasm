/**
 * Type definitions for Eigen.wasm
 * Copyright (c) 2008-2024 Eigen Authors
 * Copyright (c) 2025 Superstruct Ltd, New Zealand
 * Licensed under Mozilla Public License v2.0
 */

export interface EigenOptions {
  /** Enable SIMD optimizations (default: true) */
  simdOptimizations?: boolean
  /** Enable WebGPU acceleration for large matrices (default: false) */
  webgpuAcceleration?: boolean
  /** Maximum memory in MB (default: 512) */
  maxMemoryMB?: number
  /** Enable debug logging (default: false) */
  debug?: boolean
}

export interface Matrix {
  /** Number of rows */
  rows: number
  /** Number of columns */
  cols: number
  /** Matrix data in column-major order */
  data: Float64Array | Float32Array
}

export interface Vector {
  /** Vector size */
  size: number
  /** Vector data */
  data: Float64Array | Float32Array
}

export interface MatrixResult<T = Float64Array> {
  /** Operation success status */
  success: boolean
  /** Result matrix data */
  data?: T
  /** Result dimensions */
  rows?: number
  cols?: number
  /** Error message if operation failed */
  error?: string
}

export interface DecompositionResult {
  /** Decomposition success status */
  success: boolean
  /** L matrix (for LU) or Q matrix (for QR) */
  L?: Matrix
  /** U matrix (for LU) or R matrix (for QR) */
  U?: Matrix
  /** Permutation matrix (for LU with pivoting) */
  P?: Matrix
  /** Singular values (for SVD) */
  singularValues?: Vector
  /** Left singular vectors (for SVD) */
  leftVectors?: Matrix
  /** Right singular vectors (for SVD) */
  rightVectors?: Matrix
  /** Error message if decomposition failed */
  error?: string
}

export interface EigenvalueResult {
  /** Computation success status */
  success: boolean
  /** Real parts of eigenvalues */
  eigenvalues?: Vector
  /** Eigenvectors (columns are eigenvectors) */
  eigenvectors?: Matrix
  /** Error message if computation failed */
  error?: string
}

export interface SolverResult {
  /** Solver success status */
  success: boolean
  /** Solution vector */
  solution?: Vector
  /** Relative error of solution */
  relativeError?: number
  /** Error message if solving failed */
  error?: string
}

export interface PerformanceMetrics {
  /** Matrix multiplication time (ms) */
  matrixMultiply: number
  /** LU decomposition time (ms) */
  luDecomposition: number
  /** Eigenvalue computation time (ms) */
  eigenvalueComputation: number
  /** SVD computation time (ms) */
  svdComputation: number
}

export interface MemoryStats {
  /** Currently allocated memory (MB) */
  allocatedMB: number
  /** Peak memory usage (MB) */
  peakMB: number
}

/** Decomposition types for matrix_decompose */
export enum DecompositionType {
  LU = 0,
  QR = 1,
  SVD = 2,
}

/** Matrix creation options */
export interface MatrixCreationOptions {
  /** Fill with zeros (default), ones, or random values */
  fill?: 'zeros' | 'ones' | 'random' | 'identity'
  /** Random seed for reproducible random matrices */
  seed?: number
  /** Use single precision (float32) instead of double precision */
  singlePrecision?: boolean
}

/** Geometry types for 3D operations */
export interface EulerAngles {
  x: number
  y: number
  z: number
}

export interface Quaternion {
  w: number
  x: number
  y: number
  z: number
}

export interface RotationMatrix extends Matrix {
  rows: 3
  cols: 3
}

/** Performance profiling results */
export interface ProfileResults {
  /** Matrix size tested */
  matrixSize: number
  /** Number of iterations */
  iterations: number
  /** Results for different operations */
  results: {
    /** Matrix multiplication (ms) */
    multiply: number
    /** LU decomposition (ms) */
    lu: number
    /** Eigenvalue computation (ms) */
    eigen: number
    /** SVD computation (ms) */
    svd: number
  }
}