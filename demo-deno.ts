#!/usr/bin/env -S deno run --allow-read --allow-write

import EigenWasm, { DecompositionType } from "./src/lib/index.ts"

async function demo() {
  console.log("🔢 Eigen.wasm - Advanced Linear Algebra Demo")
  console.log("=" + "=".repeat(60))

  const eigen = new EigenWasm({
    simdOptimizations: true,
    webgpuAcceleration: false,
    debug: true,
    maxMemoryMB: 256,
  })

  console.log("\n🚀 Initializing Eigen.wasm...")
  await eigen.initialize()
  console.log(`✅ ${eigen.getBuildInfo()}`)

  // Basic matrix operations
  console.log("\n📊 Basic Matrix Operations")
  console.log("-".repeat(40))

  const A = eigen.createMatrix(3, 3, { fill: 'random', seed: 42 })
  const B = eigen.createMatrix(3, 3, { fill: 'random', seed: 43 })

  console.log("Matrix A (3×3):")
  printMatrix(A.data, 3, 3)

  console.log("Matrix B (3×3):")
  printMatrix(B.data, 3, 3)

  // Matrix multiplication
  const C = eigen.multiply(A, B)
  if (C.success) {
    console.log("A × B =")
    printMatrix(C.data!, 3, 3)
  }

  // Matrix transpose
  const At = eigen.transpose(A)
  if (At.success) {
    console.log("A^T =")
    printMatrix(At.data!, 3, 3)
  }

  // Matrix determinant
  const detA = eigen.determinant(A)
  console.log(`det(A) = ${detA.toFixed(6)}`)

  // Vector operations
  console.log("\n📐 Vector Operations")
  console.log("-".repeat(40))

  const v1 = eigen.createVector(5, { fill: 'random', seed: 100 })
  const v2 = eigen.createVector(5, { fill: 'random', seed: 101 })

  console.log(`Vector 1: [${Array.from(v1.data).map(x => x.toFixed(3)).join(", ")}]`)
  console.log(`Vector 2: [${Array.from(v2.data).map(x => x.toFixed(3)).join(", ")}]`)

  const dotProduct = eigen.dot(v1, v2)
  console.log(`Dot product: ${dotProduct.toFixed(6)}`)

  // SIMD comparison
  console.log("\n⚡ SIMD Performance Comparison")
  console.log("-".repeat(40))

  const size = 64
  const iterations = 100

  const simdA = eigen.createMatrix(size, size, { fill: 'random', seed: 1, singlePrecision: true })
  const simdB = eigen.createMatrix(size, size, { fill: 'random', seed: 2, singlePrecision: true })

  // SIMD benchmark
  const startSIMD = performance.now()
  for (let i = 0; i < iterations; i++) {
    const result = eigen.multiply(simdA, simdB, true)
    if (!result.success) throw new Error("SIMD multiply failed")
  }
  const endSIMD = performance.now()
  const simdTime = endSIMD - startSIMD

  // Scalar benchmark
  const startScalar = performance.now()
  for (let i = 0; i < iterations; i++) {
    const result = eigen.multiply(simdA, simdB, false)
    if (!result.success) throw new Error("Scalar multiply failed")
  }
  const endScalar = performance.now()
  const scalarTime = endScalar - startScalar

  const speedup = scalarTime / simdTime
  const ops = 2 * size * size * size * iterations
  const simdGFLOPS = (ops / 1e9) / (simdTime / 1000)
  const scalarGFLOPS = (ops / 1e9) / (scalarTime / 1000)

  console.log(`Matrix size: ${size}×${size}, Iterations: ${iterations}`)
  console.log(`SIMD:   ${simdTime.toFixed(2)}ms (${simdGFLOPS.toFixed(2)} GFLOPS)`)
  console.log(`Scalar: ${scalarTime.toFixed(2)}ms (${scalarGFLOPS.toFixed(2)} GFLOPS)`)
  console.log(`Speedup: ${speedup.toFixed(2)}x`)

  // Matrix decompositions
  console.log("\n🔍 Matrix Decompositions")
  console.log("-".repeat(40))

  const testMatrix = eigen.createMatrix(4, 4, { fill: 'random', seed: 200 })

  // Add diagonal dominance for better numerical stability
  for (let i = 0; i < 4; i++) {
    testMatrix.data[i * 4 + i] += 5.0
  }

  console.log("Test matrix (4×4):")
  printMatrix(testMatrix.data, 4, 4)

  // LU decomposition
  const luResult = eigen.decompose(testMatrix, DecompositionType.LU)
  if (luResult.success) {
    console.log("LU Decomposition successful:")
    console.log("L matrix:")
    printMatrix(luResult.L!.data, 4, 4)
    console.log("U matrix:")
    printMatrix(luResult.U!.data, 4, 4)
  }

  // QR decomposition
  const qrResult = eigen.decompose(testMatrix, DecompositionType.QR)
  if (qrResult.success) {
    console.log("QR Decomposition successful:")
    console.log("Q matrix:")
    printMatrix(qrResult.L!.data, 4, 4) // L contains Q
    console.log("R matrix (first 4 rows):")
    printMatrix(qrResult.U!.data, 4, 4) // U contains R
  }

  // SVD decomposition
  const svdMatrix = eigen.createMatrix(3, 2, { fill: 'random', seed: 300 })
  const svdResult = eigen.decompose(svdMatrix, DecompositionType.SVD)
  if (svdResult.success) {
    console.log("SVD Decomposition successful:")
    console.log(`Singular values: [${Array.from(svdResult.singularValues!.data).map(x => x.toFixed(4)).join(", ")}]`)
  }

  // Eigenvalue computation
  console.log("\n🎯 Eigenvalue Computation")
  console.log("-".repeat(40))

  // Create symmetric matrix for real eigenvalues
  const symMatrix = eigen.createMatrix(3, 3, { fill: 'random', seed: 400 })
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      const avg = (symMatrix.data[i * 3 + j] + symMatrix.data[j * 3 + i]) / 2
      symMatrix.data[i * 3 + j] = avg
      symMatrix.data[j * 3 + i] = avg
    }
  }

  console.log("Symmetric matrix:")
  printMatrix(symMatrix.data, 3, 3)

  const eigenResult = eigen.eigenvalues(symMatrix, true)
  if (eigenResult.success) {
    console.log(`Eigenvalues: [${Array.from(eigenResult.eigenvalues!.data).map(x => x.toFixed(6)).join(", ")}]`)
    console.log("Eigenvectors:")
    printMatrix(eigenResult.eigenvectors!.data, 3, 3)
  }

  // Linear system solving
  console.log("\n🧮 Linear System Solving")
  console.log("-".repeat(40))

  const sysA = eigen.createMatrix(3, 3, { fill: 'random', seed: 500 })
  // Add diagonal dominance for better conditioning
  for (let i = 0; i < 3; i++) {
    sysA.data[i * 3 + i] += 5.0
  }

  const sysB = eigen.createVector(3, { fill: 'random', seed: 501 })

  console.log("System matrix A:")
  printMatrix(sysA.data, 3, 3)
  console.log(`Right-hand side b: [${Array.from(sysB.data).map(x => x.toFixed(3)).join(", ")}]`)

  const solveResult = eigen.solve(sysA, sysB)
  if (solveResult.success) {
    console.log(`Solution x: [${Array.from(solveResult.solution!.data).map(x => x.toFixed(6)).join(", ")}]`)

    // Verify A*x ≈ b
    const verification = eigen.multiply(sysA, { rows: 3, cols: 1, data: solveResult.solution!.data })
    if (verification.success) {
      console.log(`Verification A*x: [${Array.from(verification.data!).map(x => x.toFixed(6)).join(", ")}]`)

      const residual = Array.from(verification.data!).map((val, i) => Math.abs(val - sysB.data[i]))
      const maxResidual = Math.max(...residual)
      console.log(`Maximum residual: ${maxResidual.toExponential(3)}`)
    }
  }

  // Advanced operations (if available)
  console.log("\n🚀 Advanced Operations")
  console.log("-".repeat(40))

  try {
    const rank = eigen.rank(testMatrix, 1e-12)
    console.log(`Matrix rank: ${rank}`)

    const condNum = eigen.conditionNumber(testMatrix)
    console.log(`Condition number: ${condNum.toExponential(3)}`)

    // 3D geometry
    const euler = { x: Math.PI / 4, y: Math.PI / 6, z: 0 }
    const rotMatrix = eigen.rotationFromEuler(euler)
    console.log("Rotation matrix from Euler angles (π/4, π/6, 0):")
    printMatrix(rotMatrix.data, 3, 3)

    const quat = eigen.quaternionFromRotation(rotMatrix)
    console.log(`Quaternion: [${quat.w.toFixed(4)}, ${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}]`)

    // Performance profiling
    const profile = eigen.profile(32, 5)
    console.log("\nPerformance Profile (32×32 matrices, 5 iterations):")
    console.log(`  Matrix multiply: ${profile.results.multiply.toFixed(2)}ms`)
    console.log(`  LU decomposition: ${profile.results.lu.toFixed(2)}ms`)
    console.log(`  Eigenvalues: ${profile.results.eigen.toFixed(2)}ms`)
    console.log(`  SVD: ${profile.results.svd.toFixed(2)}ms`)

  } catch (error) {
    console.log(`ℹ️  Advanced features not available: ${error.message}`)
  }

  // Memory statistics
  console.log("\n💾 Memory Statistics")
  console.log("-".repeat(40))

  const memStats = eigen.getMemoryStats()
  console.log(`Allocated: ${memStats.allocatedMB} MB`)
  console.log(`Peak usage: ${memStats.peakMB} MB`)

  console.log("\n🧹 Cleanup")
  console.log("-".repeat(40))
  eigen.cleanup()
  console.log("✅ Eigen.wasm cleanup complete")

  console.log("\n🎉 Demo completed successfully!")
}

function printMatrix(data: Float64Array | Float32Array, rows: number, cols: number, precision = 3) {
  for (let i = 0; i < rows; i++) {
    const row = []
    for (let j = 0; j < cols; j++) {
      const value = data[j * rows + i] // Column-major indexing
      row.push(value.toFixed(precision).padStart(precision + 3))
    }
    console.log(`[${row.join(" ")}]`)
  }
}

if (import.meta.main) {
  try {
    await demo()
  } catch (error) {
    console.error("❌ Demo failed:", error.message)
    console.error(error.stack)
    Deno.exit(1)
  }
}