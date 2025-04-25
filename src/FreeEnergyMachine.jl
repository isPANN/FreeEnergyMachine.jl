module FreeEnergyMachine

using SparseArrays
using LinearAlgebra
using CUDA
using Random
using Statistics
using Distributions
using Flux
using Zygote
using Enzyme

include("abstractproblem.jl")
include("max_cut.jl")
include("optimizer.jl")
include("fem_solver.jl")
include("sbm_solver.jl")

export is_binary, entropy_term, entropy_term_grad, energy_term_grad, energy_term, infer
export AdamOpt, RMSpropOpt, LinearAnnealing, ExponentialAnnealing, InverseAnnealing
export load_matrix, MaxCut
export Solver, initialize, fem_iterate, free_energy

# Export DSBM solver
export SimulatedBifurcation, SimulatedBifurcationState, simulate_bifurcation!
end
