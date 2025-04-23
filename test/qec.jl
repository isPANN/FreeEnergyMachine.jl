using FreeEnergyMachine
using Test
using FreeEnergyMachine: _even_probability, _odd_probability
using TensorQEC
using TensorQEC: generate_spin_glass_sa

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

    em = iid_error(T(0.1),T(0.1),T(0.1),d*d)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(1234)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(tanner, syd)...)
    nsweeps = 100
    prob,_ = generate_spin_glass_sa(tanner, em, collect(T, 0:1e-3:1.0), nsweeps)
end