using FreeEnergyMachine
using Test

@testset "FreeEnergyMachine.jl" begin
    include("problems.jl")
    include("dsbm_test.jl")
end
