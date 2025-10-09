using FreeEnergyMachine
using Test

@testset "FreeEnergyMachine.jl" begin
    include("problems.jl")
end

@testset "GPU" begin
    include("gpu_test.jl")
end