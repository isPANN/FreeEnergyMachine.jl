using FreeEnergyMachine
using Test
using CUDA

@testset "FreeEnergyMachine.jl" begin
    include("problems.jl")
end

if CUDA.functional()
    @testset "GPU" begin
        include("gpu_test.jl")
    end
end