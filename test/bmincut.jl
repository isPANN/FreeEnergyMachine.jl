using Test
using CUDA, cuDNN
using FreeEnergyMachine
using SparseArrays
using LinearAlgebra
using Flux

@testset "bMinCut" begin
    test_content = """
                   5 5
                   0 1 1
                   1 2
                   2 3 3
                   3 4 4
                   4 0
                  """

   test_filename = "test_graph.txt"
   open(test_filename, "w") do f
       write(f, test_content)
   end

   # test_filename = joinpath(pkgdir(FreeEnergyMachine), "FEM/benchmarks/maxcut/Gset/G12")
   A, node_num, edge_num = load_weighted_graph(test_filename; zero_based = true)
   
   @test node_num == 5
   @test edge_num == 5
   @test A[1, 2] == 1.0f0
   rm(test_filename)

   q = 3
   prob = bMinCut(A, q)
   h = randn(Float32, 5, node_num, q)  # (batch_size, node_num, q)
   p = softmax(h, dims=2)

   @show infer(prob, p)
   @test size(energy_term(prob, p)) == (5,)
   @test size(entropy_term(prob, p)) == (5,)
   @test !is_binary(prob)

   # Test with automatic differentiation
   config_auto = SolverConfig(
       betamin = 1/0.264, 
       betamax = 1/1.1e-3, 
       annealing = ExponentialAnnealing(), 
       optimizer = AdamOpt(0.01), 
       manual_grad = false, 
       h_factor = 1, 
       seed = 1234,
       device = "cuda"
   )
   solver_auto = Solver(prob, 100, 1000, q; config = config_auto)
   p_auto = fem_iterate(solver_auto)
   @show infer(prob, p_auto)

   # Test the free_energy function
   h_test = initialize(solver_auto)
   fe = free_energy(solver_auto, h_test, 1)
   @test typeof(fe) <: Real

   # Test that Zygote can differentiate the free_energy function
   grad = Zygote.gradient(h -> free_energy(solver_auto, h, 1), h_test)[1]
   @test size(grad) == size(h_test)
end

@testset "Sparse vs Dense Equivalence" begin
    # Create a small test graph
    n = 10
    I = [1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    J = [2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10]
    V = Float64[1.0, 0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.6, 1.4, 0.4]
    
    W_sparse = sparse(I, J, V, n, n)
    W_sparse = W_sparse + W_sparse'  # Make symmetric
    W_dense = Matrix(W_sparse)
    
    q = 2
    λ = 5.0
    
    # Create problems
    prob_sparse = bMinCut(W_sparse, q, λ=λ)
    prob_dense = bMinCut(W_dense, q, λ=λ)
    
    @test prob_sparse.node_num == prob_dense.node_num
    @test prob_sparse.q == prob_dense.q
    @test prob_sparse.λ ≈ prob_dense.λ
    
    # Test energy computation
    batch_size = 4
    p = rand(Float64, batch_size, n, q)
    p = p ./ sum(p, dims=3)  # Normalize
    
    E_sparse = energy_term(prob_sparse, p)
    E_dense = energy_term(prob_dense, p)
    
    @test E_sparse ≈ E_dense rtol=1e-10
    
    println("✓ Energy computation: sparse and dense match")
end

@testset "Sparse Matrix q > 2" begin
    # Test with more than 2 communities
    n = 8
    density = 0.3
    n_edges = Int(round(n * (n - 1) * density / 2))
    
    I = rand(1:n, n_edges)
    J = rand(1:n, n_edges)
    V = rand(n_edges)
    
    W_sparse = sparse(I, J, V, n, n)
    W_sparse = W_sparse + W_sparse'
    W_dense = Matrix(W_sparse)
    
    q = 4
    prob_sparse = bMinCut(W_sparse, q, λ=3.0)
    prob_dense = bMinCut(W_dense, q, λ=3.0)
    
    batch_size = 3
    p = rand(Float64, batch_size, n, q)
    p = p ./ sum(p, dims=3)
    
    E_sparse = energy_term(prob_sparse, p)
    E_dense = energy_term(prob_dense, p)
    
    @test E_sparse ≈ E_dense rtol=1e-10
    
    println("✓ q=4 communities: sparse and dense match")
end

@testset "Infer Function with Sparse" begin
    # Test inference with sparse matrices
    n = 6
    W_sparse = sparse([1,1,2,3,4], [2,3,3,4,5], Float64[1,1,1,1,1], n, n)
    W_sparse = W_sparse + W_sparse'
    W_dense = Matrix(W_sparse)
    
    q = 2
    prob_sparse = bMinCut(W_sparse, q, λ=1.0)
    prob_dense = bMinCut(W_dense, q, λ=1.0)
    
    # Create a simple configuration
    p = zeros(Float64, 2, n, q)
    p[1, 1:3, 1] .= 1.0  # First 3 nodes in community 1
    p[1, 4:6, 2] .= 1.0  # Last 3 nodes in community 2
    p[2, 1:4, 1] .= 1.0  # First 4 nodes in community 1
    p[2, 5:6, 2] .= 1.0  # Last 2 nodes in community 2
    
    cut_sparse = infer(prob_sparse, p)
    cut_dense = infer(prob_dense, p)
    
    @test cut_sparse ≈ cut_dense rtol=1e-10
    
    println("✓ Infer function: sparse and dense match")
end

@testset "is_binary Trait" begin
    n = 5
    W = sparse([1,1,2,3,4], [2,3,3,4,5], Float64[1,1,1,1,1], n, n)
    W = W + W'
    
    prob_binary = bMinCut(W, 2, λ=1.0)
    prob_multi = bMinCut(W, 4, λ=1.0)
    
    @test is_binary(prob_binary) == true
    @test is_binary(prob_multi) == false
    
    println("✓ is_binary trait: correctly based on q value")
end

@testset "Large Sparse Graph" begin
    # Test with a larger sparse graph
    n = 100
    density = 0.05
    n_edges = Int(round(n * (n - 1) * density / 2))
    
    I = rand(1:n, n_edges)
    J = rand(1:n, n_edges)
    V = rand(n_edges)
    
    W_sparse = sparse(I, J, V, n, n)
    W_sparse = W_sparse + W_sparse'
    
    q = 3
    prob = bMinCut(W_sparse, q, λ=5.0)
    
    # Test that we can compute energy without errors
    batch_size = 2
    p = rand(Float64, batch_size, n, q)
    p = p ./ sum(p, dims=3)
    
    E = energy_term(prob, p)
    
    @test length(E) == batch_size
    @test all(isfinite.(E))
    
    println("✓ Large sparse graph (n=100): computation successful")
end

@testset "Edge Cases" begin
    # Test with empty graph (no edges)
    n = 5
    W_empty = spzeros(Float64, n, n)
    
    prob = bMinCut(W_empty, 2, λ=1.0)
    
    p = rand(Float64, 2, n, 2)
    p = p ./ sum(p, dims=3)
    
    E = energy_term(prob, p)
    
    @test all(isfinite.(E))
    
    # Test with complete graph
    W_complete = sparse(ones(Float64, n, n) - I(n))
    prob_complete = bMinCut(W_complete, 2, λ=1.0)
    
    E_complete = energy_term(prob_complete, p)
    
    @test all(isfinite.(E_complete))
    
    println("✓ Edge cases: empty and complete graphs work correctly")
end

@testset "Type Stability" begin
    n = 10
    W_sparse = sparse(rand(1:n, 20), rand(1:n, 20), rand(20), n, n)
    W_sparse = W_sparse + W_sparse'
    
    prob_f64 = bMinCut(W_sparse, 2, λ=5.0)
    prob_f32 = bMinCut(Float32.(W_sparse), 2, λ=5.0f0)
    
    p_f64 = rand(Float64, 2, n, 2)
    p_f64 = p_f64 ./ sum(p_f64, dims=3)
    
    p_f32 = Float32.(p_f64)
    
    E_f64 = energy_term(prob_f64, p_f64)
    E_f32 = energy_term(prob_f32, p_f32)
    
    @test eltype(E_f64) == Float64
    @test eltype(E_f32) == Float32
    
    println("✓ Type stability: Float32 and Float64 work correctly")
end

