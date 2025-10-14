using Test
using FreeEnergyMachine
using CUDA

@testset "Device Management" begin
    @test CPU() isa AbstractDevice
    @test GPU() isa AbstractDevice
    
    # Test select_device
    @test select_device("cpu") isa CPU
    
    if CUDA.functional()
        @test select_device("cuda") isa GPU
        @test select_device("gpu") isa GPU
    end
    
    # Test array operations
    x = randn(10, 10)
    x_cpu = to_device(CPU(), x)
    @test x_cpu isa Array
    
    if CUDA.functional()
        x_gpu = to_device(GPU(), x)
        @test x_gpu isa CuArray
        
        # Test roundtrip
        x_back = to_device(CPU(), x_gpu)
        @test x_back isa Array
        @test x_back ≈ x
    end
end

# Test MaxCut problem on CPU and GPU
@testset "MaxCut Problem" begin
    # Create a simple graph
    coupling = [0.0 1.0 1.0;
                    1.0 0.0 1.0;
                    1.0 1.0 0.0]
    
    # Test CPU version
    problem_cpu = MaxCut(coupling, device="cpu")
    @test problem_cpu.node_num == 3
    @test problem_cpu.coupling isa Array
    
    # Test GPU version
    if CUDA.functional()
        CUDA.seed!(1234)

        problem_gpu = MaxCut(coupling, device="cuda")
        @test problem_gpu.node_num == 3
        @test problem_gpu.coupling isa CuArray
        
        # Test energy computation on GPU
        p_gpu = CUDA.rand(Float64, 5, 3)  # 5 trials, 3 nodes
        energy_gpu = energy_term(problem_gpu, p_gpu)
        @test energy_gpu isa CuArray
        @test length(energy_gpu) == 5
        
        # Compare with CPU results
        p_cpu = Array(p_gpu)
        energy_cpu = energy_term(problem_cpu, p_cpu)
        @test Array(energy_gpu) ≈ energy_cpu
    end
end


# Test Solver with GPU
@testset "Solver with GPU" begin
    CUDA.allowscalar(false)
    coupling = [0.0 1.0 1.0;
                    1.0 0.0 1.0;
                    1.0 1.0 0.0]
    
    # CPU solver
    problem_cpu = MaxCut(coupling, device="cpu")
    config_cpu = SolverConfig(
        betamin=0.01,
        betamax=0.5,
        manual_grad=true,
        device="cpu"
    )
    solver_cpu = Solver(problem_cpu, 10, 100; config=config_cpu)
    
    @test solver_cpu.problem === problem_cpu
    @test solver_cpu.num_trials == 10
    @test solver_cpu.num_steps == 100
    
    # Test initialization
    h_cpu = initialize(solver_cpu)
    @test h_cpu isa Array
    @test size(h_cpu) == (10, 3)
    
    # Test iteration
    p_cpu = fem_iterate(solver_cpu)
    @test p_cpu isa Array
    @test size(p_cpu) == (10, 3)
    @test all(0 .<= p_cpu .<= 1)
    
    if CUDA.functional()
        # GPU solver
        problem_gpu = MaxCut(coupling, device="cuda")
        config_gpu = SolverConfig(
            betamin=0.01,
            betamax=0.5,
            manual_grad=true,
            device="cuda"
        )
        solver_gpu = Solver(problem_gpu, 10, 100; config=config_gpu)
        
        @test solver_gpu.problem === problem_gpu
        
        # Test initialization on GPU
        h_gpu = initialize(solver_gpu)
        @test h_gpu isa CuArray
        @test size(h_gpu) == (10, 3)
        
        # Test iteration on GPU
        p_gpu = fem_iterate(solver_gpu)
        @test p_gpu isa CuArray
        @test size(p_gpu) == (10, 3)
        @test all(0 .<= Array(p_gpu) .<= 1)
    end
end

# Test inference
@testset "Inference" begin    
    CUDA.allowscalar(false)
    coupling = [0.0 1.0 1.0;
                    1.0 0.0 1.0;
                    1.0 1.0 0.0]
    
    p_cpu = [0.9 0.1 0.8;
                0.2 0.7 0.3]
    
    problem_cpu = MaxCut(coupling, device="cpu")
    result_cpu = infer(problem_cpu, p_cpu)
    @test result_cpu isa Array
    @test length(result_cpu) == 2
    
    if CUDA.functional()
        problem_gpu = MaxCut(coupling, device="cuda")
        p_gpu = CuArray(p_cpu)
        result_gpu = infer(problem_gpu, p_gpu)
        @test result_gpu isa CuArray
        @test Array(result_gpu) ≈ result_cpu
    end
end

