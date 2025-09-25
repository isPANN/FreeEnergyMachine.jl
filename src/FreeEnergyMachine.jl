module FreeEnergyMachine

using SparseArrays
using LinearAlgebra
using Random
using Statistics
using Flux
using Zygote

include("abstractproblem.jl")
include("problems/weighted_graph.jl")
# include specific problems
include("problems/max_cut.jl")
include("problems/qubo.jl")

# include solvers
include("optimizer.jl")
include("fem_solver.jl")


# load problem
export load_weighted_graph
export MaxCut
export QUBO

# determine if spins in the problem are binary
export is_binary

# calculate the energy and the energy gradient of the problem
export energy_term, energy_term_grad

# calculate the entropy and the entropy gradient of the problem
export entropy_term, entropy_term_grad

# FEM solver
export Solver
export initialize
export fem_iterate
export free_energy

# Optimizer
export AdamOpt, RMSpropOpt
export LinearAnnealing, ExponentialAnnealing, InverseAnnealing

# infer the solution of the problem
export infer
end
