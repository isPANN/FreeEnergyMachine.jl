"""
GPU Example for FreeEnergyMachine.jl

This example demonstrates how to use GPU acceleration with CUDA.jl
for solving combinatorial optimization problems.
"""

using FreeEnergyMachine
using CUDA

# Check if CUDA is available
if CUDA.functional()
    println("CUDA is available!")
    println("GPU: ", CUDA.name(CUDA.device()))
else
    println("CUDA is not available, falling back to CPU")
end

# Example 1: MaxCut on GPU
println("\n=== Example 1: MaxCut Problem ===")

# Create a simple graph (triangle)
coupling = Float64[
    0.0  1.0  1.0
    1.0  0.0  1.0
    1.0  1.0  0.0
]

# Solve on CPU
println("\nSolving on CPU...")
problem_cpu = MaxCut(coupling, device="cpu")
config_cpu = SolverConfig(
    betamin=0.01,
    betamax=0.5,
    annealing=InverseAnnealing(),
    optimizer=AdamOpt(0.1),
    manual_grad=true,
    h_factor=0.01,
    device="cpu"
)
solver_cpu = Solver(problem_cpu, 10, 1000; config=config_cpu)

@time p_cpu = fem_iterate(solver_cpu)
E_cpu = infer(problem_cpu, p_cpu)
println("CPU Results - Energy: ", minimum(E_cpu))
println("Best configuration: ", round.(p_cpu[argmax(E_cpu), :]))

# Solve on GPU (if available)
if CUDA.functional()
    println("\nSolving on GPU...")
    problem_gpu = MaxCut(coupling, device="cuda")
    config_gpu = SolverConfig(
        betamin=0.01,
        betamax=0.5,
        annealing=InverseAnnealing(),
        optimizer=AdamOpt(0.1),
        manual_grad=true,
        h_factor=0.01,
        device="cuda"
    )
    solver_gpu = Solver(problem_gpu, 10, 1000; config=config_gpu)
    
    @time p_gpu = fem_iterate(solver_gpu)
    E_gpu = infer(problem_gpu, p_gpu)
    
    # Transfer results back to CPU for printing
    E_gpu_cpu = Array(E_gpu)
    p_gpu_cpu = Array(p_gpu)
    
    println("GPU Results - Energy: ", minimum(E_gpu_cpu))
    println("Best configuration: ", round.(p_gpu_cpu[argmax(E_gpu_cpu), :]))
end

# Example 2: Larger MaxCut Problem
println("\n=== Example 2: Larger MaxCut Problem ===")

# Create a random graph
n_nodes = 50
coupling_large = Float64.(rand(n_nodes, n_nodes))
coupling_large = (coupling_large + coupling_large') / 2  # Make symmetric
for i in 1:n_nodes
    coupling_large[i, i] = 0.0
end

if CUDA.functional()
    println("\nSolving larger problem on GPU...")
    problem_gpu = MaxCut(coupling_large, device="cuda")
    config_gpu = SolverConfig(
        betamin=0.01,
        betamax=0.5,
        annealing=InverseAnnealing(),
        optimizer=AdamOpt(0.1),
        manual_grad=true,
        h_factor=0.01,
        seed=42,
        device="cuda"
    )
    solver_gpu = Solver(problem_gpu, 100, 1000; config=config_gpu)
    
    @time p_gpu = fem_iterate(solver_gpu)
    E_gpu = infer(problem_gpu, p_gpu)
    
    E_gpu_cpu = Array(E_gpu)
    println("Best energy found: ", maximum(E_gpu_cpu))
    println("Average energy: ", sum(E_gpu_cpu) / length(E_gpu_cpu))
end

