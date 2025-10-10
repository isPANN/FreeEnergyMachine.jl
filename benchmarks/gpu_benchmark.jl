"""
GPU vs CPU Benchmark for FreeEnergyMachine.jl

This script compares the performance of CPU and GPU implementations
for various problem sizes and configurations.
"""

using FreeEnergyMachine
using CUDA
using Printf
using Statistics

struct BenchmarkResult
    device::String
    problem_size::Int
    num_trials::Int
    num_steps::Int
    time_seconds::Float64
    memory_mb::Float64
    best_energy::Float64
    avg_energy::Float64
end

function run_maxcut_benchmark(
    n::Int, 
    num_trials::Int, 
    num_steps::Int;
    device::String="cpu",
    manual_grad::Bool=true
)
    # Create random coupling matrix
    coupling = Float64.(rand(n, n))
    coupling = (coupling + coupling') / 2  # Make symmetric
    for i in 1:n
        coupling[i, i] = 0.0
    end
    
    # Create problem
    problem = MaxCut(coupling, device=device)
    
    # Configure solver
    config = SolverConfig(
        betamin=0.01,
        betamax=0.5,
        annealing=InverseAnnealing(),
        optimizer=AdamOpt(0.1),
        manual_grad=manual_grad,
        h_factor=0.01,
        seed=42,
        device=device
    )
    
    # Create solver
    solver = Solver(problem, num_trials, num_steps; config=config)
    
    # Warm-up run (important for GPU)
    if device == "cuda" || device == "gpu"
        _ = fem_iterate(solver)
        CUDA.synchronize()
        GC.gc()
    end
    
    # Benchmark run
    # Note: Memory tracking is simplified as CUDA.jl API varies by version
    if device == "cuda" || device == "gpu"
        CUDA.reclaim()  # Clean up memory pool
        GC.gc()
    end
    
    start_time = time()
    p = fem_iterate(solver)
    
    if device == "cuda" || device == "gpu"
        CUDA.synchronize()  # Ensure all GPU operations complete
    end
    
    elapsed = time() - start_time
    
    # Memory usage estimation (optional, may vary by CUDA.jl version)
    memory_used = 0.0  # Simplified: focus on time performance
    
    # Get results
    E = infer(problem, p)
    E_cpu = device in ["cuda", "gpu"] ? Array(E) : E
    
    return BenchmarkResult(
        device,
        n,
        num_trials,
        num_steps,
        elapsed,
        memory_used,
        maximum(E_cpu),
        mean(E_cpu)
    )
end

function print_result(result::BenchmarkResult)
    @printf("%-6s | N=%4d | Trials=%4d | Steps=%4d | Time=%7.3fs | Mem=%7.1f MB | Best E=%8.2f | Avg E=%8.2f\n",
        result.device,
        result.problem_size,
        result.num_trials,
        result.num_steps,
        result.time_seconds,
        result.memory_mb,
        result.best_energy,
        result.avg_energy
    )
end

function print_comparison(cpu_result::BenchmarkResult, gpu_result::BenchmarkResult)
    speedup = cpu_result.time_seconds / gpu_result.time_seconds
    @printf("\nSpeedup: %.2fx (GPU is %.2fx faster than CPU)\n", speedup, speedup)
    @printf("Time saved: %.2f seconds\n", cpu_result.time_seconds - gpu_result.time_seconds)
end

# Main benchmark
println("="^100)
println("FreeEnergyMachine.jl GPU Benchmark")
println("="^100)

# Check GPU availability
if !CUDA.functional()
    println("\nWARNING: CUDA is not available. Only CPU benchmarks will be run.")
    println("To enable GPU benchmarks, ensure you have:")
    println("  - NVIDIA GPU with CUDA support")
    println("  - CUDA driver installed")
    println("  - CUDA.jl properly configured")
    println()
else
    println("\nGPU Information:")
    println("  Device: ", CUDA.name(CUDA.device()))
    println("  Total Memory: ", CUDA.totalmem(CUDA.device()) ÷ 1024^2, " MB")
    # println("  Free Memory: ", CUDA.available_memory(CUDA.device()) ÷ 1024^2, " MB")
    println()
end

# Benchmark configurations
configs = [
    (n=20, trials=10, steps=500),
    (n=50, trials=20, steps=500),
    (n=100, trials=50, steps=1000),
    (n=200, trials=100, steps=1000),
    (n=500, trials=200, steps=1000),
]

println("\nRunning Benchmarks...")
println("-"^100)

results_cpu = BenchmarkResult[]
results_gpu = BenchmarkResult[]

for (i, config) in enumerate(configs)
    println("\nBenchmark $(i)/$(length(configs)): N=$(config.n), Trials=$(config.trials), Steps=$(config.steps)")
    println("-"^100)
    
    # CPU benchmark
    print("Running CPU... ")
    flush(stdout)
    result_cpu = run_maxcut_benchmark(
        config.n, config.trials, config.steps; 
        device="cpu", manual_grad=true
    )
    push!(results_cpu, result_cpu)
    print_result(result_cpu)
    
    # GPU benchmark (if available)
    if CUDA.functional()
        print("Running GPU... ")
        flush(stdout)
        try
            result_gpu = run_maxcut_benchmark(
                config.n, config.trials, config.steps; 
                device="cuda", manual_grad=true
            )
            push!(results_gpu, result_gpu)
            print_result(result_gpu)
            print_comparison(result_cpu, result_gpu)
        catch e
            println("ERROR: ", e)
            println("Skipping GPU benchmark for this configuration")
        end
    end
end

# Summary
println("\n" * "="^100)
println("Summary")
println("="^100)

println("\nCPU Results:")
println("-"^100)
println("Problem Size | Trials | Steps | Time (s) | Best Energy | Avg Energy")
println("-"^100)
for r in results_cpu
    @printf("%12d | %6d | %5d | %8.3f | %11.2f | %10.2f\n",
        r.problem_size, r.num_trials, r.num_steps, 
        r.time_seconds, r.best_energy, r.avg_energy
    )
end

if !isempty(results_gpu)
    println("\nGPU Results:")
    println("-"^100)
    println("Problem Size | Trials | Steps | Time (s) | Memory (MB) | Best Energy | Avg Energy")
    println("-"^100)
    for r in results_gpu
        @printf("%12d | %6d | %5d | %8.3f | %11.1f | %11.2f | %10.2f\n",
            r.problem_size, r.num_trials, r.num_steps, 
            r.time_seconds, r.memory_mb, r.best_energy, r.avg_energy
        )
    end
    
    println("\nSpeedup Summary:")
    println("-"^100)
    println("Problem Size | Trials | CPU Time | GPU Time | Speedup")
    println("-"^100)
    for (cpu_r, gpu_r) in zip(results_cpu, results_gpu)
        speedup = cpu_r.time_seconds / gpu_r.time_seconds
        @printf("%12d | %6d | %8.3fs | %8.3fs | %6.2fx\n",
            cpu_r.problem_size, cpu_r.num_trials,
            cpu_r.time_seconds, gpu_r.time_seconds, speedup
        )
    end
end

println("\n" * "="^100)
println("Benchmark Complete!")
println("="^100)

