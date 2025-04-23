using FreeEnergyMachine
using Test
using Flux
# using Zygote
using FreeEnergyMachine: get_betas

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
    @test size(energy_term(prob, p)) == (5,)
    @test size(entropy_term(prob, p)) == (5,)
    @test size(entropy_term_grad(prob, p)) == (5, node_num)

    num_steps = 1000
    betamin = 1/0.264
    betamax = 1/1.1e-3
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
