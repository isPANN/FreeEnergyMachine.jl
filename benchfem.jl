using FreeEnergyMachine
using Test
using Flux
using FreeEnergyMachine: get_betas
using Random


@testset "femqec" begin
    T = Float64
    d = 5

    em = iid_error(T(0.1),T(0.1),T(0.1),d*d)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(2354)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    Eopt, config = optimal_energy(em,tanner,syd)
    prob = generate_femqec(tanner, em,config)

    num_steps = 100000
    betamin = T(1e-5)
    betamax = T(1)
    betas = get_betas(InverseAnnealing(), num_steps, betamin, betamax)

    nbatch = 10
    solver = Solver(nbatch, num_steps, betas; h_factor =1, flavor = 2)

    # Iterate through the solver with manual gradient
    p_manual = fem_iterate(prob,solver)

    E = energy_term(prob,p_manual)
    minE,pos = findmax(E)

    @show E
    @show minE
    @show Eopt
end