using FreeEnergyMachine
using Random
using Profile
using ProfileView
using BenchmarkTools

# 创建一个中等规模的测试问题
function create_test_problem(n=100)
    # 创建随机图的邻接矩阵
    Random.seed!(1234)
    A = sprand(Float32, n, n, 0.1)  # 10% 连接密度
    A = A + A'  # 对称化
    A = A - Diagonal(A)  # 移除自环
    return MaxCut(A; discretization = true)
end

# 基准测试函数
function benchmark_fem(prob, num_trials=50, num_steps=500)
    config = SolverConfig(
        betamin = 0.01,
        betamax = 0.5,
        annealing = ExponentialAnnealing(),
        optimizer = AdamOpt(0.01),
        manual_grad = true,
        h_factor = 0.01,
        seed = 1234
    )
    
    solver = Solver(prob, num_trials, num_steps, 2; config = config)
    
    # 预热
    fem_iterate(solver)
    
    # 基准测试
    println("=== 基准测试 ===")
    @time result = fem_iterate(solver)
    
    # 详细基准测试
    println("\n=== 详细基准测试 ===")
    @btime fem_iterate($solver)
    
    return result
end

# Profiling 函数
function profile_fem(prob, num_trials=50, num_steps=500)
    config = SolverConfig(
        betamin = 0.01,
        betamax = 0.5,
        annealing = ExponentialAnnealing(),
        optimizer = AdamOpt(0.01),
        manual_grad = true,
        h_factor = 0.01,
        seed = 1234
    )
    
    solver = Solver(prob, num_trials, num_steps, 2; config = config)
    
    # 预热
    fem_iterate(solver)
    
    println("=== 开始 Profiling ===")
    Profile.clear()
    @profile begin
        for i in 1:10  # 运行多次以获得更好的统计
            fem_iterate(solver)
        end
    end
    
    println("=== Profile 结果 ===")
    Profile.print(maxdepth=15)
    
    # 保存 profile 结果
    try
        ProfileView.view()
        println("Profile 可视化已打开")
    catch e
        println("无法打开 ProfileView: $e")
    end
    
    return solver
end

# 内存分配分析
function analyze_allocations(prob, num_trials=50, num_steps=100)
    config = SolverConfig(
        betamin = 0.01,
        betamax = 0.5,
        annealing = ExponentialAnnealing(),
        optimizer = AdamOpt(0.01),
        manual_grad = true,
        h_factor = 0.01,
        seed = 1234
    )
    
    solver = Solver(prob, num_trials, num_steps, 2; config = config)
    
    # 预热
    fem_iterate(solver)
    
    println("=== 内存分配分析 ===")
    @time @allocated fem_iterate(solver)
    
    # 详细内存分析
    allocs = @allocated fem_iterate(solver)
    println("总内存分配: $(allocs / 1024^2) MB")
    
    return solver
end

# 比较手动梯度 vs 自动微分
function compare_gradient_methods(prob, num_trials=20, num_steps=100)
    println("=== 梯度方法比较 ===")
    
    # 手动梯度
    config_manual = SolverConfig(
        betamin = 0.01,
        betamax = 0.5,
        annealing = ExponentialAnnealing(),
        optimizer = AdamOpt(0.01),
        manual_grad = true,
        h_factor = 0.01,
        seed = 1234
    )
    solver_manual = Solver(prob, num_trials, num_steps, 2; config = config_manual)
    
    # 自动微分
    config_auto = SolverConfig(
        betamin = 0.01,
        betamax = 0.5,
        annealing = ExponentialAnnealing(),
        optimizer = AdamOpt(0.01),
        manual_grad = false,
        h_factor = 0.01,
        seed = 1234
    )
    solver_auto = Solver(prob, num_trials, num_steps, 2; config = config_auto)
    
    # 预热
    fem_iterate(solver_manual)
    fem_iterate(solver_auto)
    
    println("手动梯度:")
    @btime fem_iterate($solver_manual)
    
    println("自动微分:")
    @btime fem_iterate($solver_auto)
end

function main()
    println("开始性能分析...")
    
    # 创建不同规模的测试问题
    problems = [
        ("小规模 (n=50)", create_test_problem(50)),
        ("中规模 (n=100)", create_test_problem(100)),
        ("大规模 (n=200)", create_test_problem(200))
    ]
    
    for (name, prob) in problems
        println("\n" * "="^50)
        println("测试问题: $name")
        println("节点数: $(prob.node_num)")
        println("="^50)
        
        # 基准测试
        benchmark_fem(prob)
        
        # 梯度方法比较
        compare_gradient_methods(prob)
        
        # 内存分析
        analyze_allocations(prob)
    end
    
    # 详细 profiling (仅对中规模问题)
    println("\n" * "="^50)
    println("详细 Profiling (中规模问题)")
    println("="^50)
    prob = create_test_problem(100)
    profile_fem(prob)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

