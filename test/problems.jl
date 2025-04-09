@testset "Max-Cut" begin
    using Flux
    using Zygote

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

    prob = MaxCut(node_num, edge_num, A; discretization = true, dtype = Float32)
    h = randn(Float32, 5, node_num)  # (batch_size, node_num)
    p = sigmoid.(h)
    # @show round.(p)
    @show infer(prob, p)
    @test size(p) == size(energy_term_grad(prob, p)) == (5, node_num)
    @test size(energy_term(prob, p)) == (5,)
    @test size(entropy_term(prob, p)) == (5,)
    @test size(entropy_term_grad(prob, p)) == (5, node_num)
    @test is_binary(prob)

    solver = Solver(prob, 10, 1000; betamin = 1/0.264, betamax = 1/1.1e-3, annealing = InverseAnnealing(), optimizer_type = AdamOpt(0.01), manual_grad = true, h_factor =1, q = 2, seed = 1234)
    @test solver.problem == prob
    @test solver.learning_rate ≈ 0.01f0

    # Initialize the solver
    h = initialize(solver)
    @test size(h) == (solver.num_trials, node_num)

    # Iterate through the solver with manual gradient
    p_manual = fem_iterate(solver)
    @show infer(prob, p_manual)

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
