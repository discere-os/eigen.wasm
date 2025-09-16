/**
 * Eigen.wasm - WebAssembly port of Eigen C++ linear algebra library
 * Copyright (c) 2008-2024 Eigen Authors
 * Copyright (c) 2025 Superstruct Ltd, New Zealand
 * Licensed under Mozilla Public License v2.0
 */

import type {
  EigenOptions,
  Matrix,
  Vector,
  MatrixResult,
  DecompositionResult,
  EigenvalueResult,
  SolverResult,
  PerformanceMetrics,
  MemoryStats,
  MatrixCreationOptions,
  EulerAngles,
  Quaternion,
  RotationMatrix,
  ProfileResults,
} from './types.ts'
import { DecompositionType } from './types.ts'

export * from './types.ts'

export default class EigenWasm {
  private module: any = null
  private initialized = false
  private options: EigenOptions

  // Cached function pointers for performance
  private _matrixMultiply: any = null
  private _matrixMultiplySIMD: any = null
  private _vectorDot: any = null
  private _vectorDotSIMD: any = null
  private _vectorDotSingle: any = null
  private _matrixInvert: any = null
  private _matrixDecompose: any = null
  private _eigenvalues: any = null
  private _matrixSolve: any = null
  private _matrixTranspose: any = null
  private _matrixDeterminant: any = null
  private _benchmarkMatrixMultiply: any = null
  private _matrixRank: any = null
  private _matrixConditionNumber: any = null
  private _matrixPseudoinverse: any = null
  private _matrixExponential: any = null
  private _rotationMatrixFromEuler: any = null
  private _quaternionFromRotationMatrix: any = null
  private _profileOperations: any = null
  private _memoryStats: any = null
  private _enableSIMD: any = null
  private _webgpuAvailable: any = null
  private _eigenBuildInfo: any = null

  constructor(options: EigenOptions = {}) {
    this.options = {
      simdOptimizations: true,
      webgpuAcceleration: false,
      maxMemoryMB: 512,
      debug: false,
      ...options,
    }
  }

  async initialize(): Promise<void> {
    if (this.initialized) return

    const wasmBinary = await this.loadWasmBinary()
    const moduleFactory = await this.loadModuleFactory()

    this.module = await moduleFactory({
      wasmBinary,
      locateFile: (path: string) => {
        if (path.endsWith('.wasm')) {
          return new URL('../../install/wasm/' + path, import.meta.url).href
        }
        return path
      },
    })

    // Setup function bindings
    this.setupBindings()

    // Configure SIMD if supported
    if (this.options.simdOptimizations && this._enableSIMD) {
      this._enableSIMD(true)
    }

    // Check WebGPU availability
    if (this.options.webgpuAcceleration && this._webgpuAvailable) {
      const gpuAvailable = this._webgpuAvailable()
      if (!gpuAvailable && this.options.debug) {
        console.warn('WebGPU requested but not available')
      }
    }

    this.initialized = true

    if (this.options.debug) {
      console.log('Eigen.wasm initialized:', this.getBuildInfo())
    }
  }

  private async loadWasmBinary(): Promise<ArrayBuffer> {
    if (typeof globalThis.Deno !== 'undefined') {
      // Deno environment
      const wasmPath = new URL('../../install/wasm/eigen-main.wasm', import.meta.url).pathname
      const wasmBuffer = await Deno.readFile(wasmPath)
      return wasmBuffer.buffer
    }

    // Browser environment - try CDN
    const cdnUrl = 'https://wasm.discere.cloud/npm/@discere-os/eigen.wasm/main.wasm'
    const response = await fetch(cdnUrl)
    if (!response.ok) {
      throw new Error(`Failed to load Eigen.wasm: ${response.statusText}`)
    }
    return await response.arrayBuffer()
  }

  private async loadModuleFactory(): Promise<any> {
    const modulePath = new URL('../../install/wasm/eigen-main.js', import.meta.url).href
    const module = await import(modulePath)
    return module.default || module.EigenModule
  }

  private setupBindings(): void {
    const { cwrap } = this.module

    // Core matrix operations
    this._matrixMultiply = cwrap('matrix_multiply', 'number', ['number', 'number', 'number', 'number', 'number', 'number'])
    this._matrixMultiplySIMD = cwrap('matrix_multiply_simd', 'number', ['number', 'number', 'number', 'number', 'number', 'number'])
    this._vectorDot = cwrap('vector_dot', 'number', ['number', 'number', 'number'])
    this._vectorDotSIMD = cwrap('vector_dot_simd', 'number', ['number', 'number', 'number'])

    // Try to bind vector_dot_single, fallback to SIMD version if not available
    try {
      this._vectorDotSingle = cwrap('vector_dot_single', 'number', ['number', 'number', 'number'])
    } catch (e) {
      console.warn('vector_dot_single not available, using SIMD version for non-SIMD calls')
      this._vectorDotSingle = this._vectorDotSIMD
    }
    this._matrixInvert = cwrap('matrix_invert', 'number', ['number', 'number', 'number'])
    this._matrixDecompose = cwrap('matrix_decompose', 'number', ['number', 'number', 'number', 'number', 'number', 'number', 'number'])
    this._eigenvalues = cwrap('eigenvalues', 'number', ['number', 'number', 'number', 'number'])
    this._matrixSolve = cwrap('matrix_solve', 'number', ['number', 'number', 'number', 'number'])
    this._matrixTranspose = cwrap('matrix_transpose', 'number', ['number', 'number', 'number', 'number'])
    this._matrixDeterminant = cwrap('matrix_determinant', 'number', ['number', 'number'])

    // Advanced operations (MAIN_MODULE only)
    try {
      this._benchmarkMatrixMultiply = cwrap('benchmark_matrix_multiply', 'number', ['number', 'number'])
      this._matrixRank = cwrap('matrix_rank', 'number', ['number', 'number', 'number', 'number'])
      this._matrixConditionNumber = cwrap('matrix_condition_number', 'number', ['number', 'number'])
      this._matrixPseudoinverse = cwrap('matrix_pseudoinverse', 'number', ['number', 'number', 'number', 'number', 'number'])
      this._matrixExponential = cwrap('matrix_exponential', 'number', ['number', 'number', 'number'])
      this._rotationMatrixFromEuler = cwrap('rotation_matrix_from_euler', 'number', ['number', 'number'])
      this._quaternionFromRotationMatrix = cwrap('quaternion_from_rotation_matrix', 'number', ['number', 'number'])
      this._profileOperations = cwrap('profile_operations', 'void', ['number', 'number', 'number'])
    } catch (e) {
      if (this.options.debug) {
        console.warn('Some advanced features not available (SIDE_MODULE build)', e)
      }
    }

    // Utility functions
    this._memoryStats = cwrap('memory_stats', 'number', ['number', 'number'])
    this._enableSIMD = cwrap('enable_simd', 'number', ['number'])
    this._webgpuAvailable = cwrap('webgpu_available_check', 'number', [])

    try {
      this._eigenBuildInfo = cwrap('eigen_build_info', 'string', [])
    } catch (e) {
      // Not available in SIDE_MODULE
    }
  }

  // Matrix creation utilities
  createMatrix(rows: number, cols: number, options: MatrixCreationOptions = {}): Matrix {
    const singlePrecision = options.singlePrecision ?? false
    const data = singlePrecision ? new Float32Array(rows * cols) : new Float64Array(rows * cols)

    switch (options.fill) {
      case 'ones':
        data.fill(1.0)
        break
      case 'random':
        const seed = options.seed ?? Math.random() * 1000000
        let rng = seed
        for (let i = 0; i < data.length; i++) {
          rng = (rng * 9301 + 49297) % 233280
          data[i] = (rng / 233280.0) * 2.0 - 1.0 // Random in [-1, 1]
        }
        break
      case 'identity':
        if (rows === cols) {
          for (let i = 0; i < rows; i++) {
            data[i * rows + i] = 1.0
          }
        }
        break
      case 'zeros':
      default:
        data.fill(0.0)
        break
    }

    return { rows, cols, data }
  }

  createVector(size: number, options: MatrixCreationOptions = {}): Vector {
    const matrix = this.createMatrix(size, 1, options)
    return { size, data: matrix.data }
  }

  // Core matrix operations
  multiply(a: Matrix, b: Matrix, useSIMD = this.options.simdOptimizations): MatrixResult {
    if (a.cols !== b.rows) {
      return { success: false, error: `Matrix dimensions incompatible: ${a.rows}×${a.cols} * ${b.rows}×${b.cols}` }
    }

    const resultData = new (a.data.constructor as any)(a.rows * b.cols)
    const aPtr = this.allocateMatrix(a.data)
    const bPtr = this.allocateMatrix(b.data)
    const resultPtr = this.allocateMatrix(resultData)

    try {
      const func = useSIMD && a.data instanceof Float32Array ? this._matrixMultiplySIMD : this._matrixMultiply
      const result = func(aPtr, bPtr, resultPtr, a.rows, a.cols, b.cols)

      if (result === 0) {
        // Copy result back
        const resultHeap = this.getHeapArray(resultPtr, resultData.length, a.data instanceof Float32Array)
        resultData.set(resultHeap)

        return {
          success: true,
          data: resultData,
          rows: a.rows,
          cols: b.cols,
        }
      } else {
        return { success: false, error: 'Matrix multiplication failed' }
      }
    } finally {
      this.module._free(aPtr)
      this.module._free(bPtr)
      this.module._free(resultPtr)
    }
  }

  dot(a: Vector, b: Vector, useSIMD = this.options.simdOptimizations): number {
    if (a.size !== b.size) {
      throw new Error(`Vector dimensions incompatible: ${a.size} vs ${b.size}`)
    }

    const aPtr = this.allocateMatrix(a.data)
    const bPtr = this.allocateMatrix(b.data)

    try {
      // Choose appropriate function based on precision and SIMD preference
      if (a.data instanceof Float32Array) {
        const func = useSIMD ? this._vectorDotSIMD : this._vectorDotSingle
        if (!func) {
          throw new Error(`Vector dot function not available: useSIMD=${useSIMD}`)
        }
        return func(aPtr, bPtr, a.size)
      } else {
        // For double precision, only non-SIMD is available
        if (!this._vectorDot) {
          throw new Error('Vector dot function not available for double precision')
        }
        return this._vectorDot(aPtr, bPtr, a.size)
      }
    } finally {
      this.module._free(aPtr)
      this.module._free(bPtr)
    }
  }

  invert(matrix: Matrix): MatrixResult {
    if (matrix.rows !== matrix.cols) {
      return { success: false, error: 'Matrix must be square for inversion' }
    }

    const resultData = new (matrix.data.constructor as any)(matrix.data.length)
    const matPtr = this.allocateMatrix(matrix.data)
    const resultPtr = this.allocateMatrix(resultData)

    try {
      const result = this._matrixInvert(matPtr, resultPtr, matrix.rows)

      if (result === 0) {
        const resultHeap = this.getHeapArray(resultPtr, resultData.length, matrix.data instanceof Float32Array)
        resultData.set(resultHeap)

        return {
          success: true,
          data: resultData,
          rows: matrix.rows,
          cols: matrix.cols,
        }
      } else if (result === -2) {
        return { success: false, error: 'Matrix is not invertible (singular)' }
      } else {
        return { success: false, error: 'Matrix inversion failed' }
      }
    } finally {
      this.module._free(matPtr)
      this.module._free(resultPtr)
    }
  }

  decompose(matrix: Matrix, type: DecompositionType): DecompositionResult {
    const matPtr = this.allocateMatrix(matrix.data)

    try {
      switch (type) {
        case DecompositionType.LU: {
          const lData = new (matrix.data.constructor as any)(matrix.rows * Math.min(matrix.rows, matrix.cols))
          const uData = new (matrix.data.constructor as any)(Math.min(matrix.rows, matrix.cols) * matrix.cols)

          const lPtr = this.allocateMatrix(lData)
          const uPtr = this.allocateMatrix(uData)

          try {
            const result = this._matrixDecompose(matPtr, matrix.rows, matrix.cols, type, lPtr, uPtr, 0)

            if (result === 0) {
              const lHeap = this.getHeapArray(lPtr, lData.length, matrix.data instanceof Float32Array)
              const uHeap = this.getHeapArray(uPtr, uData.length, matrix.data instanceof Float32Array)

              lData.set(lHeap)
              uData.set(uHeap)

              return {
                success: true,
                L: { rows: matrix.rows, cols: Math.min(matrix.rows, matrix.cols), data: lData },
                U: { rows: Math.min(matrix.rows, matrix.cols), cols: matrix.cols, data: uData },
              }
            } else {
              return { success: false, error: 'LU decomposition failed' }
            }
          } finally {
            this.module._free(lPtr)
            this.module._free(uPtr)
          }
        }

        case DecompositionType.QR: {
          const qData = new (matrix.data.constructor as any)(matrix.rows * matrix.rows)
          const rData = new (matrix.data.constructor as any)(matrix.rows * matrix.cols)

          const qPtr = this.allocateMatrix(qData)
          const rPtr = this.allocateMatrix(rData)

          try {
            const result = this._matrixDecompose(matPtr, matrix.rows, matrix.cols, type, qPtr, rPtr, 0)

            if (result === 0) {
              const qHeap = this.getHeapArray(qPtr, qData.length, matrix.data instanceof Float32Array)
              const rHeap = this.getHeapArray(rPtr, rData.length, matrix.data instanceof Float32Array)

              qData.set(qHeap)
              rData.set(rHeap)

              return {
                success: true,
                L: { rows: matrix.rows, cols: matrix.rows, data: qData }, // Q matrix
                U: { rows: matrix.rows, cols: matrix.cols, data: rData }, // R matrix
              }
            } else {
              return { success: false, error: 'QR decomposition failed' }
            }
          } finally {
            this.module._free(qPtr)
            this.module._free(rPtr)
          }
        }

        case DecompositionType.SVD: {
          const uData = new (matrix.data.constructor as any)(matrix.rows * matrix.rows)
          const vData = new (matrix.data.constructor as any)(matrix.cols * matrix.cols)
          const sData = new (matrix.data.constructor as any)(Math.min(matrix.rows, matrix.cols))

          const uPtr = this.allocateMatrix(uData)
          const vPtr = this.allocateMatrix(vData)
          const sPtr = this.allocateMatrix(sData)

          try {
            const result = this._matrixDecompose(matPtr, matrix.rows, matrix.cols, type, uPtr, vPtr, sPtr)

            if (result === 0) {
              const uHeap = this.getHeapArray(uPtr, uData.length, matrix.data instanceof Float32Array)
              const vHeap = this.getHeapArray(vPtr, vData.length, matrix.data instanceof Float32Array)
              const sHeap = this.getHeapArray(sPtr, sData.length, matrix.data instanceof Float32Array)

              uData.set(uHeap)
              vData.set(vHeap)
              sData.set(sHeap)

              return {
                success: true,
                leftVectors: { rows: matrix.rows, cols: matrix.rows, data: uData },
                rightVectors: { rows: matrix.cols, cols: matrix.cols, data: vData },
                singularValues: { size: Math.min(matrix.rows, matrix.cols), data: sData },
              }
            } else {
              return { success: false, error: 'SVD decomposition failed' }
            }
          } finally {
            this.module._free(uPtr)
            this.module._free(vPtr)
            this.module._free(sPtr)
          }
        }

        default:
          return { success: false, error: 'Unknown decomposition type' }
      }
    } finally {
      this.module._free(matPtr)
    }
  }

  eigenvalues(matrix: Matrix, computeEigenvectors = false): EigenvalueResult {
    if (matrix.rows !== matrix.cols) {
      return { success: false, error: 'Matrix must be square for eigenvalue computation' }
    }

    const eigenvalData = new (matrix.data.constructor as any)(matrix.rows)
    const eigenvecData = computeEigenvectors ? new (matrix.data.constructor as any)(matrix.data.length) : null

    const matPtr = this.allocateMatrix(matrix.data)
    const eigenvalPtr = this.allocateMatrix(eigenvalData)
    const eigenvecPtr = eigenvecData ? this.allocateMatrix(eigenvecData) : 0

    try {
      const result = this._eigenvalues(matPtr, eigenvalPtr, eigenvecPtr, matrix.rows)

      if (result === 0) {
        const eigenvalHeap = this.getHeapArray(eigenvalPtr, eigenvalData.length, matrix.data instanceof Float32Array)
        eigenvalData.set(eigenvalHeap)

        const eigenvalResult: EigenvalueResult = {
          success: true,
          eigenvalues: { size: matrix.rows, data: eigenvalData },
        }

        if (computeEigenvectors && eigenvecData && eigenvecPtr) {
          const eigenvecHeap = this.getHeapArray(eigenvecPtr, eigenvecData.length, matrix.data instanceof Float32Array)
          eigenvecData.set(eigenvecHeap)
          eigenvalResult.eigenvectors = { rows: matrix.rows, cols: matrix.rows, data: eigenvecData }
        }

        return eigenvalResult
      } else if (result === -2) {
        return { success: false, error: 'Eigenvalue computation did not converge' }
      } else {
        return { success: false, error: 'Eigenvalue computation failed' }
      }
    } finally {
      this.module._free(matPtr)
      this.module._free(eigenvalPtr)
      if (eigenvecPtr) this.module._free(eigenvecPtr)
    }
  }

  solve(A: Matrix, b: Vector): SolverResult {
    if (A.rows !== A.cols) {
      return { success: false, error: 'Coefficient matrix must be square' }
    }
    if (A.rows !== b.size) {
      return { success: false, error: `Incompatible dimensions: ${A.rows}×${A.cols} system with ${b.size}-element vector` }
    }

    const solutionData = new (b.data.constructor as any)(b.size)
    const aPtr = this.allocateMatrix(A.data)
    const bPtr = this.allocateMatrix(b.data)
    const xPtr = this.allocateMatrix(solutionData)

    try {
      const result = this._matrixSolve(aPtr, bPtr, xPtr, A.rows)

      if (result === 0) {
        const solutionHeap = this.getHeapArray(xPtr, solutionData.length, b.data instanceof Float32Array)
        solutionData.set(solutionHeap)

        return {
          success: true,
          solution: { size: b.size, data: solutionData },
          relativeError: 0.0, // Could be computed by verifying A*x - b
        }
      } else if (result === -2) {
        return { success: false, error: 'Poor solution quality (ill-conditioned matrix)' }
      } else {
        return { success: false, error: 'Linear system solving failed' }
      }
    } finally {
      this.module._free(aPtr)
      this.module._free(bPtr)
      this.module._free(xPtr)
    }
  }

  transpose(matrix: Matrix): MatrixResult {
    const resultData = new (matrix.data.constructor as any)(matrix.data.length)
    const matPtr = this.allocateMatrix(matrix.data)
    const resultPtr = this.allocateMatrix(resultData)

    try {
      const result = this._matrixTranspose(matPtr, resultPtr, matrix.rows, matrix.cols)

      if (result === 0) {
        const resultHeap = this.getHeapArray(resultPtr, resultData.length, matrix.data instanceof Float32Array)
        resultData.set(resultHeap)

        return {
          success: true,
          data: resultData,
          rows: matrix.cols,
          cols: matrix.rows,
        }
      } else {
        return { success: false, error: 'Matrix transpose failed' }
      }
    } finally {
      this.module._free(matPtr)
      this.module._free(resultPtr)
    }
  }

  determinant(matrix: Matrix): number {
    if (matrix.rows !== matrix.cols) {
      throw new Error('Matrix must be square for determinant computation')
    }

    const matPtr = this.allocateMatrix(matrix.data)

    try {
      return this._matrixDeterminant(matPtr, matrix.rows)
    } finally {
      this.module._free(matPtr)
    }
  }

  // Advanced operations (MAIN_MODULE only)
  rank(matrix: Matrix, tolerance = 1e-12): number {
    if (!this._matrixRank) {
      throw new Error('Matrix rank computation not available in SIDE_MODULE build')
    }

    const matPtr = this.allocateMatrix(matrix.data)

    try {
      return this._matrixRank(matPtr, matrix.rows, matrix.cols, tolerance)
    } finally {
      this.module._free(matPtr)
    }
  }

  conditionNumber(matrix: Matrix): number {
    if (!this._matrixConditionNumber) {
      throw new Error('Condition number computation not available in SIDE_MODULE build')
    }

    if (matrix.rows !== matrix.cols) {
      throw new Error('Matrix must be square for condition number computation')
    }

    const matPtr = this.allocateMatrix(matrix.data)

    try {
      return this._matrixConditionNumber(matPtr, matrix.rows)
    } finally {
      this.module._free(matPtr)
    }
  }

  // Geometry operations
  rotationFromEuler(euler: EulerAngles): RotationMatrix {
    if (!this._rotationMatrixFromEuler) {
      throw new Error('Rotation matrix computation not available in SIDE_MODULE build')
    }

    const eulerData = new Float64Array([euler.x, euler.y, euler.z])
    const rotationData = new Float64Array(9) // 3x3 matrix

    const eulerPtr = this.allocateMatrix(eulerData)
    const rotationPtr = this.allocateMatrix(rotationData)

    try {
      const result = this._rotationMatrixFromEuler(eulerPtr, rotationPtr)

      if (result === 0) {
        const rotationHeap = this.getHeapArray(rotationPtr, rotationData.length, false)
        rotationData.set(rotationHeap)

        return { rows: 3, cols: 3, data: rotationData } as RotationMatrix
      } else {
        throw new Error('Rotation matrix computation failed')
      }
    } finally {
      this.module._free(eulerPtr)
      this.module._free(rotationPtr)
    }
  }

  quaternionFromRotation(rotation: RotationMatrix): Quaternion {
    if (!this._quaternionFromRotationMatrix) {
      throw new Error('Quaternion computation not available in SIDE_MODULE build')
    }

    if (rotation.rows !== 3 || rotation.cols !== 3) {
      throw new Error('Rotation matrix must be 3×3')
    }

    const quatData = new Float64Array(4) // w, x, y, z

    const rotationPtr = this.allocateMatrix(rotation.data)
    const quatPtr = this.allocateMatrix(quatData)

    try {
      const result = this._quaternionFromRotationMatrix(rotationPtr, quatPtr)

      if (result === 0) {
        const quatHeap = this.getHeapArray(quatPtr, quatData.length, false)
        quatData.set(quatHeap)

        return {
          w: quatData[0],
          x: quatData[1],
          y: quatData[2],
          z: quatData[3],
        }
      } else {
        throw new Error('Quaternion computation failed')
      }
    } finally {
      this.module._free(rotationPtr)
      this.module._free(quatPtr)
    }
  }

  // Performance and diagnostics
  benchmarkMultiply(size: number, iterations = 100): number {
    if (!this._benchmarkMatrixMultiply) {
      throw new Error('Benchmarking not available in SIDE_MODULE build')
    }

    return this._benchmarkMatrixMultiply(size, iterations)
  }

  profile(matrixSize: number, iterations = 10): ProfileResults {
    if (!this._profileOperations) {
      throw new Error('Profiling not available in SIDE_MODULE build')
    }

    const resultsData = new Float64Array(4)
    const resultsPtr = this.allocateMatrix(resultsData)

    try {
      this._profileOperations(matrixSize, iterations, resultsPtr)

      const resultsHeap = this.getHeapArray(resultsPtr, resultsData.length, false)
      resultsData.set(resultsHeap)

      return {
        matrixSize,
        iterations,
        results: {
          multiply: resultsData[0],
          lu: resultsData[1],
          eigen: resultsData[2],
          svd: resultsData[3],
        },
      }
    } finally {
      this.module._free(resultsPtr)
    }
  }

  getMemoryStats(): MemoryStats {
    if (!this._memoryStats || !this.module.HEAP32) {
      return { allocatedMB: 0, peakMB: 0 }
    }

    const allocatedPtr = this.module._malloc(4)
    const peakPtr = this.module._malloc(4)

    try {
      this._memoryStats(allocatedPtr, peakPtr)

      const allocatedMB = this.module.HEAP32[allocatedPtr >> 2] || 0
      const peakMB = this.module.HEAP32[peakPtr >> 2] || 0

      return { allocatedMB, peakMB }
    } catch (e) {
      console.warn("Failed to get memory stats:", e)
      return { allocatedMB: 0, peakMB: 0 }
    } finally {
      this.module._free(allocatedPtr)
      this.module._free(peakPtr)
    }
  }

  getBuildInfo(): string {
    if (!this._eigenBuildInfo) {
      return 'Eigen.wasm build info not available (SIDE_MODULE)'
    }

    return this._eigenBuildInfo()
  }

  // Utility methods
  private allocateMatrix(data: Float64Array | Float32Array): number {
    const ptr = this.module._malloc(data.length * data.BYTES_PER_ELEMENT)
    if (data instanceof Float64Array) {
      this.module.HEAPF64.set(data, ptr >> 3)
    } else {
      this.module.HEAPF32.set(data, ptr >> 2)
    }
    return ptr
  }

  private getHeapArray(ptr: number, length: number, isFloat32: boolean): Float64Array | Float32Array {
    if (isFloat32) {
      return this.module.HEAPF32.subarray(ptr >> 2, (ptr >> 2) + length)
    } else {
      return this.module.HEAPF64.subarray(ptr >> 3, (ptr >> 3) + length)
    }
  }

  isInitialized(): boolean {
    return this.initialized
  }

  cleanup(): void {
    if (this.module) {
      this.module = null
      this.initialized = false
    }
  }
}