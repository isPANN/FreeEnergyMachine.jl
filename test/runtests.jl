using FreeEnergyMachine
using Test

@testset "FreeEnergyMachine.jl" begin
    include("problems.jl")
    include("dsbm_test.jl")
    # include("mlp_energy_test.jl")
end
