using Test
using CUDA, cuDNN
using FreeEnergyMachine
using BenchmarkTools
using Random
using Zygote

@testset "entropy" begin
    Random.seed!(1234)
    p = rand(Float32, 10, 10, 2)
    res = FreeEnergyMachine._entropy_q(p)
    # test the autodiff
    grad = Zygote.jacobian(FreeEnergyMachine._entropy_q, p)

    if CUDA.functional()
        select_device("cuda")
        p_gpu = to_device(GPU(), p)
        res_gpu = FreeEnergyMachine._entropy_q(p_gpu)
        @test Array(res_gpu) ≈ Array(res)

        grad_gpu = Zygote.jacobian(FreeEnergyMachine._entropy_q, p_gpu)
    end
end

@testset "entropy_binary" begin
    Random.seed!(1234)
    p = rand(Float32, 100, 100)
    res = FreeEnergyMachine._entropy_binary(p)
    dierect = - sum(p .* log.(p) .+ (1 .- p) .* log.(1 .- p), dims=2)
    @test res ≈ vec(dierect)
    # test the autodiff
    grad = Zygote.jacobian(FreeEnergyMachine._entropy_binary, p)

    if CUDA.functional()
        select_device("cuda")
        p_gpu = to_device(GPU(), p)
        res_gpu = FreeEnergyMachine._entropy_binary(p_gpu)
        @test Array(res_gpu) ≈ Array(res)

        grad_gpu = Zygote.jacobian(FreeEnergyMachine._entropy_binary, p_gpu)
    end    
end


@testset "energy_maxcut" begin
    Random.seed!(1234)
    W = rand(Float32, 10, 10)
    W_sym = (W .+ W') .* Float32(0.5)
    W_sym = W_sym .- Diagonal(diag(W_sym))

    p = rand(Float32, 5, 10)  #(b, N)
    problem = MaxCut(W_sym)
    E = FreeEnergyMachine.energy_term(problem, p)
    @show E
end

@testset "energy_bmincut" begin
    Random.seed!(1234)
    W = rand(Float32, 10, 10)
    W_sym = (W .+ W') .* Float32(0.5)
    W_sym = W_sym .- Diagonal(diag(W_sym))

    p = rand(Float32, 5, 10, 2)
    problem = bMinCut(W_sym, 2, λ=0.1f0)
    E = FreeEnergyMachine.energy_term(problem, p)
    @show E
    infer_E = FreeEnergyMachine.infer(problem, p)
    @show infer_E
end