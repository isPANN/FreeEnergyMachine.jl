using FreeEnergyMachine
using CUDA, cuDNN
using Printf
using Statistics
using Random

struct BenchmarkResult
    device::String
    problem_size::Int
    num_trials::Int
    num_steps::Int
    num_problems::Int
    time_mean::Float64
    time_std::Float64
    time_min::Float64
    time_max::Float64
    memory_mb::Float64
    best_energy_mean::Float64
    avg_energy_mean::Float64
end

function run_single_problem(
    n::Int, 
    num_trials::Int, 
    num_steps::Int;
    device::String="cpu",
    manual_grad::Bool=true,
    seed::Int=42
)
    # Create random coupling matrix with specific seed
    Random.seed!(seed)
    if CUDA.functional()
        CUDA.seed!(seed)
    end

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
        seed=seed,
        device=device
    )
    
    # Create solver
    solver = Solver(problem, num_trials, num_steps; config=config)
    
    # Accurate timing measurement
    if device == "cuda" || device == "gpu"
        # Ensure all previous operations are complete before starting timer
        CUDA.synchronize()
        # Use CUDA.@elapsed for accurate GPU timing
        elapsed = CUDA.@elapsed begin
            p = fem_iterate(solver)
        end
    else
        # CPU timing
        elapsed = @elapsed begin
            p = fem_iterate(solver)
        end
    end
    
    # Get results
    E = infer(problem, p)
    E_cpu = device in ["cuda", "gpu"] ? Array(E) : E
    
    return elapsed, maximum(E_cpu), mean(E_cpu)
end

function run_maxcut_benchmark(
    n::Int, 
    num_trials::Int, 
    num_steps::Int,
    num_problems::Int;
    device::String="cpu",
    manual_grad::Bool=true
)
    times = Float64[]
    best_energies = Float64[]
    avg_energies = Float64[]
    
    # Warm-up run (important for GPU)
    if device == "cuda" || device == "gpu"
        run_single_problem(n, num_trials, num_steps; device=device, manual_grad=manual_grad, seed=0)
        CUDA.synchronize()
    end
    
    # Run multiple problem instances
    for i in 1:num_problems
        elapsed, best_E, avg_E = run_single_problem(
            n, num_trials, num_steps; 
            device=device, manual_grad=manual_grad, seed=42+i
        )
        push!(times, elapsed)
        push!(best_energies, best_E)
        push!(avg_energies, avg_E)
    end
    
    # Memory usage estimation (optional, may vary by CUDA.jl version)
    memory_used = 0.0  # Simplified: focus on time performance
    
    return BenchmarkResult(
        device,
        n,
        num_trials,
        num_steps,
        num_problems,
        mean(times),
        std(times),
        minimum(times),
        maximum(times),
        memory_used,
        mean(best_energies),
        mean(avg_energies)
    )
end

function print_result(result::BenchmarkResult)
    @printf("%-6s | N=%4d | Trials=%4d | Steps=%4d | Problems=%2d | Time=%7.3f±%.3fs [%.3f-%.3f] | Best E=%8.2f | Avg E=%8.2f\n",
        result.device,
        result.problem_size,
        result.num_trials,
        result.num_steps,
        result.num_problems,
        result.time_mean,
        result.time_std,
        result.time_min,
        result.time_max,
        result.best_energy_mean,
        result.avg_energy_mean
    )
end

function print_comparison(cpu_result::BenchmarkResult, gpu_result::BenchmarkResult)
    speedup = cpu_result.time_mean / gpu_result.time_mean
    @printf("\nSpeedup: %.2fx (GPU is %.2fx faster than CPU)\n", speedup, speedup)
    @printf("Time saved: %.2f±%.2f seconds\n", 
            cpu_result.time_mean - gpu_result.time_mean,
            sqrt(cpu_result.time_std^2 + gpu_result.time_std^2))
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
    CUDA.device!(1)
    println("\nGPU Information:")
    println("  Device: ", CUDA.name(CUDA.device()))
    println("  Total Memory: ", CUDA.totalmem(CUDA.device()) ÷ 1024^2, " MB")
    println()
end

# Benchmark configurations
configs = [
    (n=100, trials=100, steps=1000, num_problems=5),
    (n=200, trials=100, steps=1000, num_problems=5),
    (n=500, trials=200, steps=1000, num_problems=5),
    (n=1000, trials=200, steps=1000, num_problems=5),
]

println("\nRunning Benchmarks...")
println("-"^100)

results_cpu = BenchmarkResult[]
results_gpu = BenchmarkResult[]

for (i, config) in enumerate(configs)
    println("\nBenchmark $(i)/$(length(configs)): N=$(config.n), Trials=$(config.trials), Steps=$(config.steps), Problems=$(config.num_problems)")
    println("-"^100)
    
    # CPU benchmark
    print("Running CPU ($(config.num_problems) different problems)... ")
    flush(stdout)
    result_cpu = run_maxcut_benchmark(
        config.n, config.trials, config.steps, config.num_problems; 
        device="cpu", manual_grad=true
    )
    push!(results_cpu, result_cpu)
    print_result(result_cpu)
    
    # GPU benchmark (if available)
    if CUDA.functional()
        print("Running GPU ($(config.num_problems) different problems)... ")
        flush(stdout)
        try
            result_gpu = run_maxcut_benchmark(
                config.n, config.trials, config.steps, config.num_problems; 
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
println("Problem Size | Trials | Steps | Problems | Time (mean±std) | Best Energy | Avg Energy")
println("-"^100)
for r in results_cpu
    @printf("%12d | %6d | %5d | %8d | %7.3f±%.3f | %11.2f | %10.2f\n",
        r.problem_size, r.num_trials, r.num_steps, r.num_problems,
        r.time_mean, r.time_std, r.best_energy_mean, r.avg_energy_mean
    )
end

if !isempty(results_gpu)
    println("\nGPU Results:")
    println("-"^100)
    println("Problem Size | Trials | Steps | Problems | Time (mean±std) | Best Energy | Avg Energy")
    println("-"^100)
    for r in results_gpu
        @printf("%12d | %6d | %5d | %8d | %7.3f±%.3f | %11.2f | %10.2f\n",
            r.problem_size, r.num_trials, r.num_steps, r.num_problems,
            r.time_mean, r.time_std, r.best_energy_mean, r.avg_energy_mean
        )
    end
    
    println("\nSpeedup Summary:")
    println("-"^100)
    println("Problem Size | Trials | Problems | CPU Time (mean±std) | GPU Time (mean±std) | Speedup")
    println("-"^100)
    for (cpu_r, gpu_r) in zip(results_cpu, results_gpu)
        speedup = cpu_r.time_mean / gpu_r.time_mean
        @printf("%12d | %6d | %8d | %7.3f±%.3f | %7.3f±%.3f | %6.2fx\n",
            cpu_r.problem_size, cpu_r.num_trials, cpu_r.num_problems,
            cpu_r.time_mean, cpu_r.time_std, 
            gpu_r.time_mean, gpu_r.time_std, speedup
        )
    end
end

println("\n" * "="^100)
println("Benchmark Complete!")
println("="^100)

