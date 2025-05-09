@testset "DSBM Solver" begin

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
    # test_filename = joinpath(pkgdir(FreeEnergyMachine), "FEM/benchmarks/maxcut/Gset/G13")

    J, node_num, edge_num = load_matrix(test_filename; zero_based = true)

    prob = MaxCut(node_num, edge_num, J; discretization = true, dtype = Float32)

    sys = SimulatedBifurcation{:dSB}(prob)

    for i in 1:10
        state = SimulatedBifurcationState(node_num, 0.1; dtype=Float32)

        state, history = simulate_bifurcation!(state, sys; nsteps=2000, dt=1.25)
        @show infer(prob, permutedims(sign.(state.x)))
    end
    
    rm(test_filename)
end
