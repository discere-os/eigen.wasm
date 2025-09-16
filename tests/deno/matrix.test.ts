import { assert, assertEquals, assertExists } from "@std/assert"
import EigenWasm, { DecompositionType } from "../../src/lib/index.ts"

Deno.test("LU decomposition", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test 3x3 matrix decomposition
  const matrix = eigen.createMatrix(3, 3)
  // Set matrix to [[2, 1, 1], [4, 3, 3], [8, 7, 9]] in column-major
  matrix.data[0] = 2; matrix.data[1] = 4; matrix.data[2] = 8  // Column 0
  matrix.data[3] = 1; matrix.data[4] = 3; matrix.data[5] = 7  // Column 1
  matrix.data[6] = 1; matrix.data[7] = 3; matrix.data[8] = 9  // Column 2

  const result = eigen.decompose(matrix, DecompositionType.LU)

  assert(result.success)
  assertExists(result.L)
  assertExists(result.U)

  // Check dimensions
  assertEquals(result.L!.rows, 3)
  assertEquals(result.L!.cols, 3)
  assertEquals(result.U!.rows, 3)
  assertEquals(result.U!.cols, 3)

  // Verify L is lower triangular (approximately)
  assert(Math.abs(result.L!.data[3]) < 1e-10) // L[0,1]
  assert(Math.abs(result.L!.data[6]) < 1e-10) // L[0,2]
  assert(Math.abs(result.L!.data[7]) < 1e-10) // L[1,2]

  // Verify U is upper triangular (approximately)
  assert(Math.abs(result.U!.data[1]) < 1e-10) // U[1,0]
  assert(Math.abs(result.U!.data[2]) < 1e-10) // U[2,0]
  assert(Math.abs(result.U!.data[5]) < 1e-10) // U[2,1]

  eigen.cleanup()
})

Deno.test("QR decomposition", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  const matrix = eigen.createMatrix(3, 2)
  // Set matrix to [[1, 1], [1, 0], [0, 1]] in column-major
  matrix.data[0] = 1; matrix.data[1] = 1; matrix.data[2] = 0  // Column 0
  matrix.data[3] = 1; matrix.data[4] = 0; matrix.data[5] = 1  // Column 1

  const result = eigen.decompose(matrix, DecompositionType.QR)

  assert(result.success)
  assertExists(result.L) // Q matrix
  assertExists(result.U) // R matrix

  // Check dimensions
  assertEquals(result.L!.rows, 3) // Q is 3x3
  assertEquals(result.L!.cols, 3)
  assertEquals(result.U!.rows, 3) // R is 3x2
  assertEquals(result.U!.cols, 2)

  // Verify R is upper triangular for the relevant part
  assert(Math.abs(result.U!.data[1]) < 1e-10) // R[1,0]
  assert(Math.abs(result.U!.data[2]) < 1e-10) // R[2,0]
  assert(Math.abs(result.U!.data[5]) < 1e-10) // R[2,1]

  eigen.cleanup()
})

Deno.test("SVD decomposition", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  const matrix = eigen.createMatrix(3, 2)
  // Set matrix to [[3, 2], [2, 3], [2, 2]] in column-major
  matrix.data[0] = 3; matrix.data[1] = 2; matrix.data[2] = 2  // Column 0
  matrix.data[3] = 2; matrix.data[4] = 3; matrix.data[5] = 2  // Column 1

  const result = eigen.decompose(matrix, DecompositionType.SVD)

  assert(result.success)
  assertExists(result.leftVectors) // U matrix
  assertExists(result.rightVectors) // V matrix
  assertExists(result.singularValues) // Sigma values

  // Check dimensions
  assertEquals(result.leftVectors!.rows, 3)
  assertEquals(result.leftVectors!.cols, 3)
  assertEquals(result.rightVectors!.rows, 2)
  assertEquals(result.rightVectors!.cols, 2)
  assertEquals(result.singularValues!.size, 2)

  // Singular values should be positive and in descending order
  assert(result.singularValues!.data[0] >= result.singularValues!.data[1])
  assert(result.singularValues!.data[0] > 0)
  assert(result.singularValues!.data[1] > 0)

  eigen.cleanup()
})

Deno.test("Eigenvalue computation", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test symmetric 2x2 matrix (should have real eigenvalues)
  const matrix = eigen.createMatrix(2, 2)
  matrix.data[0] = 4; matrix.data[2] = 1
  matrix.data[1] = 1; matrix.data[3] = 2

  // Test without eigenvectors
  const result1 = eigen.eigenvalues(matrix, false)
  assert(result1.success)
  assertExists(result1.eigenvalues)
  assertEquals(result1.eigenvalues!.size, 2)

  // Test with eigenvectors
  const result2 = eigen.eigenvalues(matrix, true)
  assert(result2.success)
  assertExists(result2.eigenvalues)
  assertExists(result2.eigenvectors)
  assertEquals(result2.eigenvalues!.size, 2)
  assertEquals(result2.eigenvectors!.rows, 2)
  assertEquals(result2.eigenvectors!.cols, 2)

  // Test identity matrix (eigenvalues should be 1, 1)
  const identity = eigen.createMatrix(3, 3, { fill: 'identity' })
  const identityResult = eigen.eigenvalues(identity, false)
  assert(identityResult.success)

  for (let i = 0; i < 3; i++) {
    assert(Math.abs(identityResult.eigenvalues!.data[i] - 1.0) < 1e-10)
  }

  eigen.cleanup()
})

Deno.test("Advanced matrix operations (MAIN_MODULE only)", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  try {
    // Test matrix rank
    const matrix = eigen.createMatrix(3, 3)
    // Create a rank-2 matrix
    matrix.data[0] = 1; matrix.data[1] = 2; matrix.data[2] = 3
    matrix.data[3] = 2; matrix.data[4] = 4; matrix.data[5] = 6
    matrix.data[6] = 0; matrix.data[7] = 0; matrix.data[8] = 1

    const rank = eigen.rank(matrix)
    assertEquals(rank, 2)

    // Test condition number
    const wellConditioned = eigen.createMatrix(3, 3, { fill: 'identity' })
    const condNum = eigen.conditionNumber(wellConditioned)
    assertEquals(condNum, 1.0)

    console.log("✅ Advanced operations available (MAIN_MODULE build)")
  } catch (error) {
    console.log("ℹ️  Advanced operations not available (SIDE_MODULE build)")
    // This is expected for SIDE_MODULE builds
    assert(error.message.includes("not available in SIDE_MODULE build"))
  }

  eigen.cleanup()
})

Deno.test("3D Geometry operations (MAIN_MODULE only)", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  try {
    // Test Euler angle to rotation matrix conversion
    const euler = { x: Math.PI / 4, y: 0, z: 0 } // 45 degrees around X axis
    const rotation = eigen.rotationFromEuler(euler)

    assertEquals(rotation.rows, 3)
    assertEquals(rotation.cols, 3)

    // For rotation around X by 45°:
    // R = [[1, 0, 0], [0, cos(45°), -sin(45°)], [0, sin(45°), cos(45°)]]
    const cos45 = Math.cos(Math.PI / 4)
    const sin45 = Math.sin(Math.PI / 4)

    assert(Math.abs(rotation.data[0] - 1.0) < 1e-10)
    assert(Math.abs(rotation.data[4] - cos45) < 1e-10)
    assert(Math.abs(rotation.data[8] - cos45) < 1e-10)

    // Test rotation matrix to quaternion conversion
    const quaternion = eigen.quaternionFromRotation(rotation)

    // For 45° rotation around X axis: q = [cos(22.5°), sin(22.5°), 0, 0]
    const halfAngle = Math.PI / 8
    const expectedW = Math.cos(halfAngle)
    const expectedX = Math.sin(halfAngle)

    assert(Math.abs(quaternion.w - expectedW) < 1e-10)
    assert(Math.abs(quaternion.x - expectedX) < 1e-10)
    assert(Math.abs(quaternion.y) < 1e-10)
    assert(Math.abs(quaternion.z) < 1e-10)

    console.log("✅ 3D geometry operations available (MAIN_MODULE build)")
  } catch (error) {
    console.log("ℹ️  3D geometry operations not available (SIDE_MODULE build)")
    assert(error.message.includes("not available in SIDE_MODULE build"))
  }

  eigen.cleanup()
})

Deno.test("Large matrix operations", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  const size = 50
  const a = eigen.createMatrix(size, size, { fill: 'random', seed: 42 })
  const b = eigen.createMatrix(size, size, { fill: 'random', seed: 123 })

  console.log(`Testing ${size}×${size} matrix multiplication...`)

  const startTime = performance.now()
  const result = eigen.multiply(a, b)
  const endTime = performance.now()

  assert(result.success)
  assertEquals(result.rows, size)
  assertEquals(result.cols, size)

  const elapsedMs = endTime - startTime
  const ops = 2 * size * size * size // Approximate FLOPs for matrix multiplication
  const gflops = (ops / 1e9) / (elapsedMs / 1000)

  console.log(`Matrix multiplication: ${elapsedMs.toFixed(2)}ms, ${gflops.toFixed(2)} GFLOPS`)

  eigen.cleanup()
})

Deno.test("Single vs double precision", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test with double precision (default)
  const doubleA = eigen.createMatrix(3, 3, { fill: 'random', seed: 42 })
  const doubleB = eigen.createMatrix(3, 3, { fill: 'random', seed: 43 })
  assert(doubleA.data instanceof Float64Array)

  const doubleResult = eigen.multiply(doubleA, doubleB)
  assert(doubleResult.success)

  // Test with single precision
  const floatA = eigen.createMatrix(3, 3, { fill: 'random', seed: 42, singlePrecision: true })
  const floatB = eigen.createMatrix(3, 3, { fill: 'random', seed: 43, singlePrecision: true })
  assert(floatA.data instanceof Float32Array)

  const floatResult = eigen.multiply(floatA, floatB, true) // Use SIMD
  assert(floatResult.success)
  assert(floatResult.data instanceof Float32Array)

  eigen.cleanup()
})