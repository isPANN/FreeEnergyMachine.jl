using FreeEnergyMachine
using Test
using CUDA, cuDNN

@testset "FreeEnergyMachine.jl" begin
    include("problems.jl")
end

@testset "calculations" begin
    include("calc.jl")
end

if CUDA.functional()
    @testset "GPU" begin
        include("gpu_test.jl")
    end
end