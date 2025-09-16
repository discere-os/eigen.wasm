import EigenWasm from "../src/lib/index.ts"

let eigen: EigenWasm

// Setup before benchmarks
await (async () => {
  eigen = new EigenWasm({ simdOptimizations: true, debug: false })
  await eigen.initialize()
  console.log("🚀 Eigen.wasm benchmark suite initialized")
})()

// Matrix multiplication benchmarks
Deno.bench("Matrix 16×16 multiply (double)", () => {
  const a = eigen.createMatrix(16, 16, { fill: 'random', seed: 42 })
  const b = eigen.createMatrix(16, 16, { fill: 'random', seed: 43 })
  eigen.multiply(a, b)
})

Deno.bench("Matrix 16×16 multiply SIMD (float)", () => {
  const a = eigen.createMatrix(16, 16, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createMatrix(16, 16, { fill: 'random', seed: 43, singlePrecision: true })
  eigen.multiply(a, b, true)
})

Deno.bench("Matrix 32×32 multiply (double)", () => {
  const a = eigen.createMatrix(32, 32, { fill: 'random', seed: 42 })
  const b = eigen.createMatrix(32, 32, { fill: 'random', seed: 43 })
  eigen.multiply(a, b)
})

Deno.bench("Matrix 32×32 multiply SIMD (float)", () => {
  const a = eigen.createMatrix(32, 32, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createMatrix(32, 32, { fill: 'random', seed: 43, singlePrecision: true })
  eigen.multiply(a, b, true)
})

Deno.bench("Matrix 64×64 multiply (double)", () => {
  const a = eigen.createMatrix(64, 64, { fill: 'random', seed: 42 })
  const b = eigen.createMatrix(64, 64, { fill: 'random', seed: 43 })
  eigen.multiply(a, b)
})

Deno.bench("Matrix 64×64 multiply SIMD (float)", () => {
  const a = eigen.createMatrix(64, 64, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createMatrix(64, 64, { fill: 'random', seed: 43, singlePrecision: true })
  eigen.multiply(a, b, true)
})

Deno.bench("Matrix 128×128 multiply (double)", () => {
  const a = eigen.createMatrix(128, 128, { fill: 'random', seed: 42 })
  const b = eigen.createMatrix(128, 128, { fill: 'random', seed: 43 })
  eigen.multiply(a, b)
})

Deno.bench("Matrix 128×128 multiply SIMD (float)", () => {
  const a = eigen.createMatrix(128, 128, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createMatrix(128, 128, { fill: 'random', seed: 43, singlePrecision: true })
  eigen.multiply(a, b, true)
})

// Vector operations benchmarks
Deno.bench("Vector dot 256 (double)", () => {
  const a = eigen.createVector(256, { fill: 'random', seed: 42 })
  const b = eigen.createVector(256, { fill: 'random', seed: 43 })
  eigen.dot(a, b)
})

Deno.bench("Vector dot 256 SIMD (float)", () => {
  const a = eigen.createVector(256, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createVector(256, { fill: 'random', seed: 43, singlePrecision: true })
  eigen.dot(a, b, true)
})

Deno.bench("Vector dot 1024 (double)", () => {
  const a = eigen.createVector(1024, { fill: 'random', seed: 42 })
  const b = eigen.createVector(1024, { fill: 'random', seed: 43 })
  eigen.dot(a, b)
})

Deno.bench("Vector dot 1024 SIMD (float)", () => {
  const a = eigen.createVector(1024, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createVector(1024, { fill: 'random', seed: 43, singlePrecision: true })
  eigen.dot(a, b, true)
})

// Matrix decomposition benchmarks
Deno.bench("LU decomposition 32×32", () => {
  const matrix = eigen.createMatrix(32, 32, { fill: 'random', seed: 42 })
  eigen.decompose(matrix, 0) // LU
})

Deno.bench("QR decomposition 32×32", () => {
  const matrix = eigen.createMatrix(32, 32, { fill: 'random', seed: 42 })
  eigen.decompose(matrix, 1) // QR
})

Deno.bench("SVD decomposition 16×16", () => {
  const matrix = eigen.createMatrix(16, 16, { fill: 'random', seed: 42 })
  eigen.decompose(matrix, 2) // SVD
})

// Matrix inversion benchmarks
Deno.bench("Matrix inversion 16×16", () => {
  const matrix = eigen.createMatrix(16, 16, { fill: 'random', seed: 42 })
  // Add identity to ensure invertibility
  for (let i = 0; i < 16; i++) {
    matrix.data[i * 16 + i] += 10.0
  }
  eigen.invert(matrix)
})

Deno.bench("Matrix inversion 32×32", () => {
  const matrix = eigen.createMatrix(32, 32, { fill: 'random', seed: 42 })
  // Add identity to ensure invertibility
  for (let i = 0; i < 32; i++) {
    matrix.data[i * 32 + i] += 10.0
  }
  eigen.invert(matrix)
})

// Linear system solving benchmarks
Deno.bench("Linear solve 32×32", () => {
  const A = eigen.createMatrix(32, 32, { fill: 'random', seed: 42 })
  const b = eigen.createVector(32, { fill: 'random', seed: 43 })
  // Add identity to ensure good conditioning
  for (let i = 0; i < 32; i++) {
    A.data[i * 32 + i] += 10.0
  }
  eigen.solve(A, b)
})

Deno.bench("Linear solve 64×64", () => {
  const A = eigen.createMatrix(64, 64, { fill: 'random', seed: 42 })
  const b = eigen.createVector(64, { fill: 'random', seed: 43 })
  // Add identity to ensure good conditioning
  for (let i = 0; i < 64; i++) {
    A.data[i * 64 + i] += 10.0
  }
  eigen.solve(A, b)
})

// Eigenvalue computation benchmarks
Deno.bench("Eigenvalues 16×16 (values only)", () => {
  const matrix = eigen.createMatrix(16, 16, { fill: 'random', seed: 42 })
  // Make symmetric for better convergence
  for (let i = 0; i < 16; i++) {
    for (let j = i + 1; j < 16; j++) {
      const val = (matrix.data[i * 16 + j] + matrix.data[j * 16 + i]) / 2
      matrix.data[i * 16 + j] = val
      matrix.data[j * 16 + i] = val
    }
  }
  eigen.eigenvalues(matrix, false)
})

Deno.bench("Eigenvalues 16×16 (with vectors)", () => {
  const matrix = eigen.createMatrix(16, 16, { fill: 'random', seed: 42 })
  // Make symmetric for better convergence
  for (let i = 0; i < 16; i++) {
    for (let j = i + 1; j < 16; j++) {
      const val = (matrix.data[i * 16 + j] + matrix.data[j * 16 + i]) / 2
      matrix.data[i * 16 + j] = val
      matrix.data[j * 16 + i] = val
    }
  }
  eigen.eigenvalues(matrix, true)
})

// Matrix creation benchmarks
Deno.bench("Create random matrix 128×128", () => {
  eigen.createMatrix(128, 128, { fill: 'random', seed: Math.random() * 1000000 })
})

Deno.bench("Create identity matrix 128×128", () => {
  eigen.createMatrix(128, 128, { fill: 'identity' })
})

Deno.bench("Create zeros matrix 256×256", () => {
  eigen.createMatrix(256, 256, { fill: 'zeros' })
})

// Cleanup after benchmarks
globalThis.addEventListener("unload", () => {
  eigen?.cleanup()
  console.log("🧹 Eigen.wasm benchmark cleanup complete")
})