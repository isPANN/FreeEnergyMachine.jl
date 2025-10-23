# Example: Using Sparse Matrices with bMinCut
# This example demonstrates how to use sparse matrices for memory-efficient
# balanced minimum cut problems, with support for GPU acceleration.

using FreeEnergyMachine
using SparseArrays
using LinearAlgebra

println("=== Sparse Matrix bMinCut Example ===\n")

# Example 1: Small sparse graph (CPU)
println("Example 1: Small sparse graph on CPU")
println("-" ^ 50)

# Create a sparse adjacency matrix (5 nodes)
I = [1, 1, 2, 2, 3, 3, 4]
J = [2, 3, 1, 3, 1, 2, 5]
V = [1.0, 0.5, 1.0, 1.5, 0.5, 1.5, 2.0]
n = 5

W_sparse = sparse(I, J, V, n, n)
W_sparse = W_sparse + W_sparse'  # Make symmetric

println("Graph size: $n nodes")
println("Number of edges: $(nnz(W_sparse) ÷ 2)")
println("Sparsity: $(1 - nnz(W_sparse) / n^2)")
println()

# Create bMinCut problem with sparse matrix
q = 2  # Binary partition
prob_sparse = bMinCut(W_sparse, q, λ=5.0, device="cpu")

println("Problem created with sparse matrix")
println("Matrix type: $(typeof(prob_sparse.coupling))")
println()

# Initialize random probabilities
batch_size = 4
p = rand(Float64, batch_size, n, q)
# Normalize to satisfy simplex constraint: sum over q = 1
p = p ./ sum(p, dims=3)

# Compute energy
E = energy_term(prob_sparse, p)
println("Energy for batch of $batch_size configurations:")
println(E)
println()

# Example 2: Larger sparse random graph
println("\nExample 2: Larger sparse random graph")
println("-" ^ 50)

# Create random sparse graph (100 nodes, ~5% density)
n_large = 100
density = 0.05
n_edges = Int(round(n_large * (n_large - 1) * density / 2))

# Generate random edges
I_large = rand(1:n_large, n_edges)
J_large = rand(1:n_large, n_edges)
V_large = rand(n_edges)

W_large = sparse(I_large, J_large, V_large, n_large, n_large)
W_large = W_large + W_large'  # Make symmetric
W_large = W_large - spdiagm(diag(W_large))  # Remove self-loops

println("Graph size: $n_large nodes")
println("Number of edges: $(nnz(W_large) ÷ 2)")
println("Sparsity: $(1 - nnz(W_large) / n_large^2)")
println("Memory footprint: ~$(Base.summarysize(W_large) ÷ 1024) KB")

# For comparison, show dense matrix memory
W_dense = Matrix(W_large)
println("Dense equivalent: ~$(Base.summarysize(W_dense) ÷ 1024) KB")
println("Memory savings: $(round(100 * (1 - Base.summarysize(W_large) / Base.summarysize(W_dense)), digits=2))%")
println()

# Create problem
q_large = 4  # 4 communities
prob_large = bMinCut(W_large, q_large, λ=10.0)

# Test with smaller batch for memory efficiency
batch_small = 2
p_large = rand(Float64, batch_small, n_large, q_large)
p_large = p_large ./ sum(p_large, dims=3)

E_large = energy_term(prob_large, p_large)
println("Energy computed successfully for large sparse graph")
println("Batch size: $batch_small, Energy: $E_large")
println()

# Example 3: Comparing dense vs sparse performance
println("\nExample 3: Performance comparison")
println("-" ^ 50)

using BenchmarkTools

n_test = 50
density_test = 0.1
n_edges_test = Int(round(n_test * (n_test - 1) * density_test / 2))

I_test = rand(1:n_test, n_edges_test)
J_test = rand(1:n_test, n_edges_test)
V_test = rand(n_edges_test)

W_test_sparse = sparse(I_test, J_test, V_test, n_test, n_test)
W_test_sparse = W_test_sparse + W_test_sparse'
W_test_dense = Matrix(W_test_sparse)

prob_test_sparse = bMinCut(W_test_sparse, 2, λ=5.0)
prob_test_dense = bMinCut(W_test_dense, 2, λ=5.0)

p_test = rand(Float64, 8, n_test, 2)
p_test = p_test ./ sum(p_test, dims=3)

println("Sparse matrix timing:")
@btime energy_term($prob_test_sparse, $p_test)

println("\nDense matrix timing:")
@btime energy_term($prob_test_dense, $p_test)

println("\n" * "=" ^ 50)
println("GPU Example (requires CUDA.jl)")
println("=" ^ 50)
println("""
# Uncomment the following to use GPU sparse matrices:
# using CUDA
# using CUDA.CUSPARSE
# 
# if CUDA.functional()
#     W_gpu = CuSparseMatrixCSC(W_sparse)
#     prob_gpu = bMinCut(W_gpu, q, λ=5.0, device="gpu")
#     p_gpu = CuArray(p)
#     E_gpu = energy_term(prob_gpu, p_gpu)
#     println("GPU computation successful!")
# end
""")

println("\n=== Example completed successfully! ===")

