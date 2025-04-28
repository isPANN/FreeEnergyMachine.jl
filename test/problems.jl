using FreeEnergyMachine
using Test
using Flux
# using Zygote
using FreeEnergyMachine: get_betas
using Enzyme
using Random

@testset "energy terms" begin
    # Test energy term for a simple MaxCut problem
    node_num = 3
    edge_num = 2
    # Simple graph with two edges: 0-1 and 1-2
    J = [0.0f0 1.0f0 0.0f0;
         1.0f0 0.0f0 1.0f0; 
         0.0f0 1.0f0 0.0f0]
    
    prob = MaxCut(node_num, edge_num, J; discretization = true)
    
    # Test with deterministic probabilities
    p = Float32[1 0 1;   # Cut value should be 2 (both edges cut)
                0 1 0;   # Cut value should be 2 (both edges cut) 
                1 1 1]   # Cut value should be 0 (no edges cut)
    
    energies = energy_term(prob, p)
    @test length(energies) == 3
    @test energies[1] ≈ -4.0f0 # Negative because energy_term returns negative cut value
    @test energies[2] ≈ -4.0f0
    @test energies[3] ≈ 0.0f0
end

@testset "overlap gradient" begin
    # Test overlap gradient computation with Enzyme
    node_num = 3
    J = Float32[0 1 0; 1 0 1; 0 1 0]
    batch_size = 2
    
    p1 = rand(Float32, batch_size, node_num)
    p2 = rand(Float32, batch_size, node_num)
    
    # Test that energy_term_grad matches numerical gradient
    function overlap_wrapper(J, h, res)
        p = sigmoid.(h)
        FreeEnergyMachine.overlap!(res, p, J, p)
        return nothing
    end
    
    h = randn(Float32, batch_size, node_num)
    gval = similar(h)
    
    res = zeros(Float32, batch_size)
    gres = fill(Float32(1), batch_size)
    gp1 = fill(Float32(1), batch_size, node_num)
    gp2 = fill(Float32(1), batch_size, node_num)
    gJ = fill(Float32(1), node_num, node_num)
    Enzyme.autodiff(Reverse, FreeEnergyMachine.overlap!, Const, Duplicated(res, gres), Duplicated(p1, gp1), Const(J), Duplicated(p2, gp2))
    Enzyme.autodiff(Reverse, overlap_wrapper, Const, Duplicated(J, gJ), Duplicated(h, gval), Duplicated(res, gres))
    
    # Test gradient has correct shape
    @test size(gval) == (batch_size, node_num)
    
    # Test gradient is not all zeros
    @test any(!iszero, gval)
end


@testset "file reading" begin
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
    A, node_num, edge_num = load_matrix(test_filename; zero_based = true)
    
    @test node_num == 5
    @test edge_num == 5
    @test A[1, 2] == 1.0f0
    rm(test_filename)
end

@testset "Max-Cut" begin
    node_num = 5
    edge_num = 5
    A = [1.0f0 1.0f0 0.0f0 0.0f0 1.0f0;
         1.0f0 0.0f0 1.0f0 0.0f0 0.0f0;
         0.0f0 1.0f0 0.0f0 3.0f0 0.0f0;
         0.0f0 0.0f0 3.0f0 0.0f0 4.0f0;
         1.0f0 0.0f0 0.0f0 4.0f0 0.0f0]
    prob = MaxCut(node_num, edge_num, A; discretization = true)
    h = randn(Float32, 5, node_num)  # (batch_size, node_num)
    p = sigmoid.(h)
    # @show round.(p)
    @show infer(prob, p)
    @test size(p) == size(energy_term_grad(prob, p)) == (5, node_num)
    @test size(FreeEnergyMachine.energy_term!(zeros(Float32, 5), prob, p)) == (5,)
    @test size(entropy_term(prob, p)) == (5,)
    @test size(entropy_term_grad(prob, p)) == (5, node_num)

    num_steps = 1000
    betamin = 1/0.264f0
    betamax = 1/1.1f-3
    betas = get_betas(InverseAnnealing(), num_steps, betamin, betamax)
    solver = Solver(10, num_steps, betas; h_factor =1, flavor = 2)

    # Initialize the solver
    h = initialize(prob, solver)
    @test size(h) == (solver.num_trials, node_num)

    # Iterate through the solver with manual gradient
    p_manual = fem_iterate(prob,solver)
    best_config, best_val = infer(prob, p_manual)
    @test best_val == 9
end

@testset "energy term grad" begin
    node_num = 3
    edge_num = 2
    # Simple graph with two edges: 0-1 and 1-2
    J = [0.0f0 1.0f0 0.0f0;
         1.0f0 0.0f0 1.0f0; 
         0.0f0 1.0f0 0.0f0]
    
    prob = MaxCut(node_num, edge_num, J; discretization = true)
    Random.seed!(1234)
    h = randn(Float32, 5, node_num)
    
    grad = energy_term_grad(prob, h)
    @show grad

    h2 = copy(h)
    delta = 1e-3
    h2[1,1] += delta
    @test ((energy_term(prob, sigmoid.(h2)) - energy_term(prob, sigmoid.(h)))/delta)[1,1] ≈ grad[1,1] atol = 1e-3
end

@testset "automatic differentiation" begin
    # Test with automatic differentiation
    solver_auto = Solver(prob, 10, 1000; betamin = 1/0.264, betamax = 1/1.1e-3, annealing = ExponentialAnnealing(), optimizer_type = AdamOpt(0.01), manual_grad = false, h_factor = 1, q = 2, seed = 1234)
    p_auto = fem_iterate(solver_auto)
    @show infer(prob, p_auto)

    # Test the free_energy function
    h_test = initialize(solver)
    fe = free_energy(solver, h_test, 1)
    @test typeof(fe) <: Real

    # Test that Zygote can differentiate the free_energy function
    grad = Zygote.gradient(h -> free_energy(solver, h, 1), h_test)[1]
    @test size(grad) == size(h_test)
end

