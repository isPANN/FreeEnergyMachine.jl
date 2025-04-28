using FreeEnergyMachine
using Test
using FreeEnergyMachine: _even_probability, generate_femqec, get_betas,optimal_energy
using TensorQEC
using TensorQEC: generate_spin_glass_sa,sa_energy
using Random
using Flux

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

    E = energy_term(prob,fill(T(0.0),1,length(prob.ops)))

    prob,_ = generate_spin_glass_sa(tanner, em, collect(T, 0:1e-3:1.0), 1000,false)
    @test sa_energy(config,prob) ≈ E[1] atol = 1e-10
end

@testset "femqec" begin
    T = Float64
    d = 3

    em = iid_error(T(0.1),T(0.1),T(0.1),d*d)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(2354)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    Eopt, config = optimal_energy(em,tanner,syd)
    prob = generate_femqec(tanner, em,config)

    num_steps = 10000
    betamin = T(1e-3)
    betamax = T(1)
    betas = get_betas(InverseAnnealing(), num_steps, betamin, betamax)
    solver = Solver(10, num_steps, betas; h_factor =1, flavor = 2)

    # Iterate through the solver with manual gradient
    p_manual = fem_iterate(prob,solver)

    E = energy_term(prob,p_manual)
    minE,pos = findmax(E)

    @test minE ≈ Eopt atol = 1e-3
    @test p_manual[pos,end] ≈ 0.0 atol = 1e-3
    @test p_manual[pos,end-1] ≈ 0.0 atol = 1e-3
end

@testset "energy term grad qec" begin
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

    h = randn(Float32, 5, 2*d*d)
    
    grad = energy_term_grad(prob, h)
    @show grad

    h2 = copy(h)
    delta = 1e-3
    h2[1,1] += delta
    @test ((energy_term(prob, sigmoid.(h2)) - energy_term(prob, sigmoid.(h)))/delta)[1,1] ≈ grad[1,1] atol = 1e-3
end

@testset "femqec" begin
    T = Float64
    tanner = CSSTannerGraph(BivariateBicycleCode(6,12, ((3,0),(0,1),(0,2)), ((0,3),(1,0),(2,0))))
    n = tanner.stgx.nq

    em = iid_error(T(0.01),T(0.01),T(0.01),n)
    Random.seed!(2354)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    Eopt, config = optimal_energy(em,tanner,syd)
    @show Eopt
    prob = generate_femqec(tanner, em,config)

    num_steps = 10000
    betamin = T(1e-5)
    betamax = T(1)
    betas = get_betas(InverseAnnealing(), num_steps, betamin, betamax)
    solver = Solver(50, num_steps, betas; h_factor =1, flavor = 2)

    # Iterate through the solver with manual gradient
    p_manual = fem_iterate(prob,solver)

    E = energy_term(prob,p_manual)
    minE,pos = findmax(E)

    @show E
    @show minE
    @show energy_term(prob,fill(T(0.0),1,length(prob.ops)))
end