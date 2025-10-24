using FreeEnergyMachine
using Test
using CUDA, cuDNN

@testset "problems" begin
    include("maxcut.jl")
    include("bmincut.jl")
end

@testset "calculations" begin
    include("calc.jl")
end

if CUDA.functional()
    @testset "GPU" begin
        include("gpu_test.jl")
    end
end