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
using TensorQEC
using TensorQEC.BitBasis
using TensorQEC: SpinGlassSA, Mod2, VecPtr,IndependentDepolarizingError,getview,_vecvec2vecptr

include("abstractproblem.jl")
include("max_cut.jl")
include("optimizer.jl")
include("fem_solver.jl")
include("sbm_solver.jl")
include("qec.jl")

export is_binary, entropy_term, entropy_term_grad, energy_term_grad, energy_term, infer
export AdamOpt, RMSpropOpt, LinearAnnealing, ExponentialAnnealing, InverseAnnealing
export load_matrix, MaxCut
export Solver, initialize, fem_iterate, free_energy

# Export DSBM solver
export SimulatedBifurcation, SimulatedBifurcationState, simulate_bifurcation!
end
