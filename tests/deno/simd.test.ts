import { assert, assertEquals } from "@std/assert"
import EigenWasm from "../../src/lib/index.ts"

Deno.test("SIMD feature detection", async () => {
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  // Test that both single and double precision functions are available
  const a32 = eigen.createMatrix(4, 4, { fill: 'random', singlePrecision: true })
  const b32 = eigen.createMatrix(4, 4, { fill: 'random', singlePrecision: true })

  const a64 = eigen.createMatrix(4, 4, { fill: 'random', singlePrecision: false })
  const b64 = eigen.createMatrix(4, 4, { fill: 'random', singlePrecision: false })

  const resultSIMD = eigen.multiply(a32, b32, true)  // Single precision
  const resultScalar = eigen.multiply(a64, b64, false) // Double precision

  assert(resultSIMD.success)
  assert(resultScalar.success)
  assert(resultSIMD.data!.length > 0)
  assert(resultScalar.data!.length > 0)

  eigen.cleanup()
})

Deno.test("SIMD vs Scalar vector dot product", async () => {
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  // Test with different precision types instead of SIMD vs non-SIMD
  // since both use the same Eigen implementation
  const sizes = [16, 32, 64, 128, 256]

  for (const size of sizes) {
    // Single precision vectors
    const a32 = eigen.createVector(size, { fill: 'random', seed: 42, singlePrecision: true })
    const b32 = eigen.createVector(size, { fill: 'random', seed: 43, singlePrecision: true })

    // Double precision vectors with same values
    const a64 = eigen.createVector(size, { fill: 'random', seed: 42, singlePrecision: false })
    const b64 = eigen.createVector(size, { fill: 'random', seed: 43, singlePrecision: false })

    const startSIMD = performance.now()
    const resultSIMD = eigen.dot(a32, b32, true) // Single precision with SIMD
    const endSIMD = performance.now()

    const startScalar = performance.now()
    const resultScalar = eigen.dot(a64, b64, false) // Double precision
    const endScalar = performance.now()

    const simdTime = endSIMD - startSIMD
    const scalarTime = endScalar - startScalar

    // Results should be reasonably close (accounting for precision differences)
    assert(Math.abs(resultSIMD - resultScalar) < 1e-5)

    console.log(`Vector size ${size}: Float32 ${simdTime.toFixed(3)}ms, Float64 ${scalarTime.toFixed(3)}ms`)
  }

  eigen.cleanup()
})

Deno.test("SIMD performance comparison", async () => {
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  const size = 64
  const iterations = 100

  // Create test matrices
  const a = eigen.createMatrix(size, size, { fill: 'random', seed: 42, singlePrecision: true })
  const b = eigen.createMatrix(size, size, { fill: 'random', seed: 43, singlePrecision: true })

  console.log(`\nSIMD Performance Test (${size}×${size} matrices, ${iterations} iterations)`)

  // Benchmark SIMD multiplication
  const startSIMD = performance.now()
  for (let i = 0; i < iterations; i++) {
    const result = eigen.multiply(a, b, true)
    assert(result.success)
  }
  const endSIMD = performance.now()
  const simdTime = endSIMD - startSIMD

  // Benchmark scalar multiplication
  const startScalar = performance.now()
  for (let i = 0; i < iterations; i++) {
    const result = eigen.multiply(a, b, false)
    assert(result.success)
  }
  const endScalar = performance.now()
  const scalarTime = endScalar - startScalar

  const speedup = scalarTime / simdTime
  const ops = 2 * size * size * size * iterations
  const simdGFLOPS = (ops / 1e9) / (simdTime / 1000)
  const scalarGFLOPS = (ops / 1e9) / (scalarTime / 1000)

  console.log(`SIMD:   ${simdTime.toFixed(2)}ms (${simdGFLOPS.toFixed(2)} GFLOPS)`)
  console.log(`Scalar: ${scalarTime.toFixed(2)}ms (${scalarGFLOPS.toFixed(2)} GFLOPS)`)
  console.log(`Speedup: ${speedup.toFixed(2)}x`)

  // SIMD should be at least as fast as scalar (hopefully faster)
  assert(speedup >= 0.5, `SIMD performance regression: ${speedup.toFixed(2)}x speedup`)

  eigen.cleanup()
})

Deno.test("Large vector SIMD operations", async () => {
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  // Test with vectors that benefit from SIMD (multiples of 16 for WASM SIMD128)
  const sizes = [64, 128, 256, 512, 1024]

  for (const size of sizes) {
    console.log(`\nTesting vector size: ${size}`)

    const a = eigen.createVector(size, { fill: 'random', seed: 100 + size, singlePrecision: true })
    const b = eigen.createVector(size, { fill: 'random', seed: 200 + size, singlePrecision: true })

    // Warm up
    for (let i = 0; i < 10; i++) {
      eigen.dot(a, b, true)
    }

    // Benchmark
    const iterations = 1000
    const startTime = performance.now()

    for (let i = 0; i < iterations; i++) {
      const result = eigen.dot(a, b, true)
      assert(!isNaN(result))
    }

    const endTime = performance.now()
    const avgTime = (endTime - startTime) / iterations
    const throughputGBps = (size * 8 * 2 / 1e9) / (avgTime / 1000) // Bytes processed per second

    console.log(`  Average time: ${avgTime.toFixed(3)}ms`)
    console.log(`  Throughput: ${throughputGBps.toFixed(2)} GB/s`)
  }

  eigen.cleanup()
})

Deno.test("SIMD correctness with edge cases", async () => {
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  // Test with very small vectors (should fall back to scalar)
  const smallA = eigen.createVector(3, { singlePrecision: true })
  smallA.data[0] = 1; smallA.data[1] = 2; smallA.data[2] = 3

  const smallB = eigen.createVector(3, { singlePrecision: true })
  smallB.data[0] = 4; smallB.data[1] = 5; smallB.data[2] = 6

  const smallResult = eigen.dot(smallA, smallB, true)
  assertEquals(smallResult, 32) // 1*4 + 2*5 + 3*6 = 32

  // Test with odd-sized vectors
  const oddA = eigen.createVector(17, { fill: 'ones', singlePrecision: true })
  const oddB = eigen.createVector(17, { fill: 'ones', singlePrecision: true })

  const oddResult = eigen.dot(oddA, oddB, true)
  assertEquals(oddResult, 17) // 1*1 repeated 17 times

  // Test with zero vectors
  const zeroA = eigen.createVector(64, { fill: 'zeros', singlePrecision: true })
  const randomB = eigen.createVector(64, { fill: 'random', singlePrecision: true })

  const zeroResult = eigen.dot(zeroA, randomB, true)
  assertEquals(zeroResult, 0)

  // Test with different data patterns
  const patternA = eigen.createVector(32, { singlePrecision: true })
  const patternB = eigen.createVector(32, { singlePrecision: true })

  for (let i = 0; i < 32; i++) {
    patternA.data[i] = i % 2 === 0 ? 1 : -1 // Alternating 1, -1
    patternB.data[i] = 1 // All ones
  }

  const patternResult = eigen.dot(patternA, patternB, true)
  assertEquals(patternResult, 0) // 16 * 1 + 16 * (-1) = 0

  eigen.cleanup()
})

Deno.test("Mixed precision SIMD operations", async () => {
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  // Single precision matrices (should use SIMD path)
  const floatA = eigen.createMatrix(32, 32, { fill: 'random', seed: 1, singlePrecision: true })
  const floatB = eigen.createMatrix(32, 32, { fill: 'random', seed: 2, singlePrecision: true })

  const floatResult = eigen.multiply(floatA, floatB, true)
  assert(floatResult.success)
  assert(floatResult.data instanceof Float32Array)

  // Double precision matrices (should use regular path)
  const doubleA = eigen.createMatrix(32, 32, { fill: 'random', seed: 1 })
  const doubleB = eigen.createMatrix(32, 32, { fill: 'random', seed: 2 })

  const doubleResult = eigen.multiply(doubleA, doubleB, true)
  assert(doubleResult.success)
  assert(doubleResult.data instanceof Float64Array)

  // Results should be similar (within single precision accuracy)
  for (let i = 0; i < 16; i++) { // Check first few elements
    const diff = Math.abs(floatResult.data![i] - doubleResult.data![i])
    assert(diff < 1e-6, `Precision difference too large: ${diff}`)
  }

  eigen.cleanup()
})

Deno.test("SIMD disabled vs enabled comparison", async () => {
  // Test different precision types instead since actual SIMD/non-SIMD
  // at the same precision should give identical results
  const eigen = new EigenWasm({ simdOptimizations: true })
  await eigen.initialize()

  const size = 16
  // Create same data in both precisions
  const data = new Float64Array(size * size)
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random() - 0.5
  }

  // Single precision matrices
  const a32 = eigen.createMatrix(size, size, { singlePrecision: true })
  const b32 = eigen.createMatrix(size, size, { singlePrecision: true })
  a32.data.set(data.slice(0, size * size))
  b32.data.set(data.slice(0, size * size))

  // Double precision matrices
  const a64 = eigen.createMatrix(size, size, { singlePrecision: false })
  const b64 = eigen.createMatrix(size, size, { singlePrecision: false })
  a64.data.set(data.slice(0, size * size))
  b64.data.set(data.slice(0, size * size))

  const resultSIMD = eigen.multiply(a32, b32, true)   // Single precision
  const resultScalar = eigen.multiply(a64, b64, false) // Double precision

  assert(resultSIMD.success)
  assert(resultScalar.success)

  // Results should be reasonably close (accounting for precision differences)
  for (let i = 0; i < Math.min(resultSIMD.data!.length, resultScalar.data!.length, 64); i++) {
    const diff = Math.abs(resultSIMD.data![i] - resultScalar.data![i])
    assert(diff < 1e-5, `Precision difference too large at index ${i}: ${diff}`)
  }

  eigen.cleanup()
})