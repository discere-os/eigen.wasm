import { assert, assertEquals, assertExists, assertThrows } from "@std/assert"
import EigenWasm from "../../src/lib/index.ts"

Deno.test("EigenWasm initialization", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  assertExists(eigen)
  assert(eigen.isInitialized())

  eigen.cleanup()
})

Deno.test("Matrix creation utilities", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test zero matrix
  const zeros = eigen.createMatrix(3, 3)
  assertEquals(zeros.rows, 3)
  assertEquals(zeros.cols, 3)
  assertEquals(zeros.data.length, 9)
  assertEquals(zeros.data[0], 0.0)

  // Test ones matrix
  const ones = eigen.createMatrix(2, 3, { fill: 'ones' })
  assertEquals(ones.rows, 2)
  assertEquals(ones.cols, 3)
  assertEquals(ones.data[0], 1.0)
  assertEquals(ones.data[5], 1.0)

  // Test identity matrix
  const identity = eigen.createMatrix(3, 3, { fill: 'identity' })
  assertEquals(identity.data[0], 1.0)  // (0,0)
  assertEquals(identity.data[1], 0.0)  // (1,0)
  assertEquals(identity.data[3], 0.0)  // (0,1)
  assertEquals(identity.data[4], 1.0)  // (1,1)
  assertEquals(identity.data[8], 1.0)  // (2,2)

  // Test random matrix
  const random = eigen.createMatrix(2, 2, { fill: 'random', seed: 12345 })
  assert(random.data[0] !== 0.0) // Should have random values
  assert(random.data[0] >= -1.0 && random.data[0] <= 1.0)

  // Test vector creation
  const vector = eigen.createVector(5, { fill: 'ones' })
  assertEquals(vector.size, 5)
  assertEquals(vector.data[0], 1.0)
  assertEquals(vector.data[4], 1.0)

  eigen.cleanup()
})

Deno.test("Matrix multiplication", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test 2x2 * 2x2 = 2x2
  const a = eigen.createMatrix(2, 2)
  a.data[0] = 1; a.data[2] = 2  // Column 0: [1, 3]
  a.data[1] = 3; a.data[3] = 4  // Column 1: [2, 4]

  const b = eigen.createMatrix(2, 2)
  b.data[0] = 5; b.data[2] = 7  // Column 0: [5, 6]
  b.data[1] = 6; b.data[3] = 8  // Column 1: [7, 8]

  const result = eigen.multiply(a, b)

  assert(result.success)
  assertEquals(result.rows, 2)
  assertEquals(result.cols, 2)
  assertExists(result.data)

  // Expected result: [[17, 23], [39, 53]] in column-major order
  assertEquals(result.data![0], 17) // (0,0)
  assertEquals(result.data![1], 39) // (1,0)
  assertEquals(result.data![2], 23) // (0,1)
  assertEquals(result.data![3], 53) // (1,1)

  // Test dimension mismatch (3x2 matrix can't be multiplied by 2x2 matrix)
  const c = eigen.createMatrix(3, 2)
  const incompatibleResult = eigen.multiply(a, c)
  console.log("Incompatible result:", incompatibleResult)
  assert(!incompatibleResult.success)
  assertExists(incompatibleResult.error)

  eigen.cleanup()
})

Deno.test("Vector dot product", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  const a = eigen.createVector(3)
  a.data[0] = 1; a.data[1] = 2; a.data[2] = 3

  const b = eigen.createVector(3)
  b.data[0] = 4; b.data[1] = 5; b.data[2] = 6

  // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  const result = eigen.dot(a, b)
  assertEquals(result, 32)

  // Test with SIMD disabled
  const resultNoSIMD = eigen.dot(a, b, false)
  assertEquals(resultNoSIMD, 32)

  // Test dimension mismatch
  const c = eigen.createVector(4)
  assertThrows(() => eigen.dot(a, c))

  eigen.cleanup()
})

Deno.test("Matrix inversion", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test 2x2 invertible matrix
  const matrix = eigen.createMatrix(2, 2)
  matrix.data[0] = 4; matrix.data[2] = 3  // Column 0: [4, 7]
  matrix.data[1] = 7; matrix.data[3] = 2  // Column 1: [3, 2]

  const invResult = eigen.invert(matrix)

  assert(invResult.success)
  assertExists(invResult.data)

  // Verify A * A^-1 ≈ I
  const identity = eigen.multiply(matrix, { rows: 2, cols: 2, data: invResult.data! })
  assert(identity.success)

  // Check diagonal elements are close to 1
  assert(Math.abs(identity.data![0] - 1.0) < 1e-10)
  assert(Math.abs(identity.data![3] - 1.0) < 1e-10)
  // Check off-diagonal elements are close to 0
  assert(Math.abs(identity.data![1]) < 1e-10)
  assert(Math.abs(identity.data![2]) < 1e-10)

  // Test singular matrix (not invertible)
  const singular = eigen.createMatrix(2, 2)
  singular.data[0] = 1; singular.data[2] = 2
  singular.data[1] = 2; singular.data[3] = 4

  const singularResult = eigen.invert(singular)
  assert(!singularResult.success)

  // Test non-square matrix
  const nonSquare = eigen.createMatrix(2, 3)
  const nonSquareResult = eigen.invert(nonSquare)
  assert(!nonSquareResult.success)

  eigen.cleanup()
})

Deno.test("Matrix transpose", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  const matrix = eigen.createMatrix(2, 3)
  // Fill with [1, 2, 3; 4, 5, 6] in column-major order
  matrix.data[0] = 1; matrix.data[1] = 4  // Column 0
  matrix.data[2] = 2; matrix.data[3] = 5  // Column 1
  matrix.data[4] = 3; matrix.data[5] = 6  // Column 2

  const result = eigen.transpose(matrix)

  assert(result.success)
  assertEquals(result.rows, 3)
  assertEquals(result.cols, 2)
  assertExists(result.data)

  // Expected: [1, 2, 3; 4, 5, 6]^T = [1, 4; 2, 5; 3, 6] in column-major
  assertEquals(result.data![0], 1) // (0,0)
  assertEquals(result.data![1], 2) // (1,0)
  assertEquals(result.data![2], 3) // (2,0)
  assertEquals(result.data![3], 4) // (0,1)
  assertEquals(result.data![4], 5) // (1,1)
  assertEquals(result.data![5], 6) // (2,1)

  eigen.cleanup()
})

Deno.test("Matrix determinant", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test 2x2 matrix with known determinant
  const matrix = eigen.createMatrix(2, 2)
  matrix.data[0] = 3; matrix.data[2] = 1
  matrix.data[1] = 4; matrix.data[3] = 2

  // Det = 3*2 - 1*4 = 6 - 4 = 2
  const det = eigen.determinant(matrix)
  assertEquals(det, 2)

  // Test 3x3 identity matrix (det = 1)
  const identity = eigen.createMatrix(3, 3, { fill: 'identity' })
  const identityDet = eigen.determinant(identity)
  assertEquals(identityDet, 1)

  // Test non-square matrix
  const nonSquare = eigen.createMatrix(2, 3)
  assertThrows(() => eigen.determinant(nonSquare))

  eigen.cleanup()
})

Deno.test("Linear system solving", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Solve Ax = b where A = [[2, 1], [1, 3]], b = [3, 4]
  const A = eigen.createMatrix(2, 2)
  A.data[0] = 2; A.data[2] = 1
  A.data[1] = 1; A.data[3] = 3

  const b = eigen.createVector(2)
  b.data[0] = 3; b.data[1] = 4

  const result = eigen.solve(A, b)

  assert(result.success)
  assertExists(result.solution)

  // Verify Ax ≈ b
  const verification = eigen.multiply(A, { rows: 2, cols: 1, data: result.solution!.data })
  assert(verification.success)
  assert(Math.abs(verification.data![0] - 3) < 1e-10)
  assert(Math.abs(verification.data![1] - 4) < 1e-10)

  // Test incompatible dimensions
  const c = eigen.createVector(3)
  const incompatibleResult = eigen.solve(A, c)
  assert(!incompatibleResult.success)

  eigen.cleanup()
})

Deno.test("Error handling", async () => {
  const eigen = new EigenWasm()
  await eigen.initialize()

  // Test operations on incompatible matrices
  const small = eigen.createMatrix(1, 1)
  const large = eigen.createMatrix(2, 3)
  const result = eigen.multiply(small, large)
  assert(!result.success)

  // Test with single precision matrices
  const floatMatrix = eigen.createMatrix(2, 2, { singlePrecision: true })
  assert(floatMatrix.data instanceof Float32Array)

  eigen.cleanup()
})

Deno.test("Memory management", async () => {
  const eigen = new EigenWasm({ debug: true })
  await eigen.initialize()

  // Perform various operations to test memory allocation/deallocation
  const matrices = []
  for (let i = 0; i < 10; i++) {
    const m = eigen.createMatrix(10, 10, { fill: 'random' })
    matrices.push(m)
  }

  // Perform operations
  for (let i = 0; i < 5; i++) {
    const result = eigen.multiply(matrices[i], matrices[i + 5])
    assert(result.success)
  }

  const stats = eigen.getMemoryStats()
  assertExists(stats)
  assert(stats.allocatedMB >= 0)
  assert(stats.peakMB >= 0)

  eigen.cleanup()
})