using FreeEnergyMachine
using Test
using FreeEnergyMachine: _even_probability, generate_femqec
using TensorQEC
using TensorQEC: generate_spin_glass_sa,sa_energy
using Random

@testset "_even_probability" begin
    pvec = [0.1,0.1]
    peven,podd = _even_probability(pvec)
    @test peven ≈ 0.82 atol = 1e-10
    @test podd ≈ 0.18 atol = 1e-10

    pvec = rand(10)
    peven,podd = _even_probability(pvec)
    @test peven + podd ≈ 1 atol = 1e-10
end

@testset "energy" begin
    T = Float32
    d = 3

    em = iid_error(T(0.2),T(0.2),T(0.2),d*d)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(2354)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(tanner, syd)...)
    config = vcat(config.xerror,config.zerror)
    prob = generate_femqec(tanner, em,config)

    E = energy_term(prob,fill(T(0.0),length(prob.ops)))

    prob,_ = generate_spin_glass_sa(tanner, em, collect(T, 0:1e-3:1.0), 1000,false)
    @test sa_energy(config,prob) ≈ E atol = 1e-10
end

@testset "Max-Cut" begin
    T = Float32
    d = 3

    em = iid_error(T(0.1),T(0.1),T(0.1),d*d)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(2354)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(tanner, syd)...)
    config = vcat(config.xerror,config.zerror)
    prob = generate_femqec(tanner, em,config)

    num_steps = 1000
    betamin = 1/0.264
    betamax = 1/1.1e-3
    betas = get_betas(InverseAnnealing(), num_steps, betamin, betamax)
    solver = Solver(10, num_steps, betas; h_factor =1, flavor = 2)

    # Initialize the solver
    h = initialize(prob, solver)

    # Iterate through the solver with manual gradient
    p_manual = fem_iterate(prob,solver)
    @show p_manual
end