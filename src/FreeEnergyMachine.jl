module FreeEnergyMachine

using SparseArrays
using LinearAlgebra
using Random
using Statistics
using Flux
using Zygote
using CUDA
using SparseArrays

include("device.jl")
include("abstractproblem.jl")
include("utils.jl")
# include specific problems
include("problems/maxcut.jl")
include("problems/bmincut.jl")
# include("problems/qubo.jl")

# include solvers
include("optimizer.jl")
include("fem_solver.jl")


# Device management
export AbstractDevice, CPU, GPU
export to_device, select_device
export array_type, create_array, randn_device
export cpu, gpu

# load problem
export load_weighted_graph
export MaxCut
export bMinCut
# export QUBO

# determine if spins in the problem are binary
export is_binary

# calculate the energy and the energy gradient of the problem
export energy_term, energy_term_grad

# calculate the entropy and the entropy gradient of the problem
export entropy_term, entropy_term_grad

# FEM solver
export Solver, SolverConfig
export initialize
export fem_iterate
export free_energy

# Optimizer
export AdamOpt, RMSpropOpt
export LinearAnnealing, ExponentialAnnealing, InverseAnnealing

# infer the solution of the problem
export infer
end
