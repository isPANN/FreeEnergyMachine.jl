module FreeEnergyMachine

using SparseArrays
using LinearAlgebra
using Random
using Statistics
using Flux
using Zygote
using JuMP

include("abstractproblem.jl")
include("max_cut.jl")
include("optimizer.jl")
include("fem_solver.jl")
include("sbm_solver.jl")
include("learning.jl")
include("mlp_energy.jl")

export is_binary, entropy_term, entropy_term_grad, energy_term_grad, energy_term, infer
export AdamOpt, RMSpropOpt, LinearAnnealing, ExponentialAnnealing, InverseAnnealing
export load_matrix, MaxCut
export Solver, initialize, fem_iterate, free_energy

# Export DSBM solver
export SimulatedBifurcation, SimulatedBifurcationState, simulate_bifurcation!

export n_bit_adder, generate_dataset, my_optimise

# Export MLP Energy Model
export MLPEnergyModel, forward, train!, predict, get_energy, get_individual_energy
end
