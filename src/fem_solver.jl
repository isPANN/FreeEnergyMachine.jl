struct Solver{P<:CombinatorialProblem, O<:OptimizerType, T<:AbstractFloat}
    problem::P
    num_trials::Int
    num_steps::Int
    betas::Vector{T}
    binary::Bool
    optimizer_type::O
    learning_rate::T
    gamma_grad::T
    manual_grad::Bool
    h_factor::T
    q::Int
    seed::Int
    dev::String # only "cpu" is supported now
    dtype::Type{T}

    function Solver(
        problem::P,
        num_trials::Int,
        num_steps::Int;
        betamin::Real = 0.01,
        betamax::Real = 0.5,
        gamma_grad::Real = 1.0,
        annealing::A = ExponentialAnnealing(),
        optimizer_type::O = AdamOpt(),
        manual_grad::Bool = false,
        h_factor::Real = 0.01,
        q::Int = 2,
        seed::Int = 1,
        device::String = "cpu"
    ) where {P<:CombinatorialProblem, A<:AnnealingStrategy, O<:OptimizerType}

        T = problem.dtype
        betas = T.(get_betas(annealing, num_steps, T(betamin), T(betamax)))

        binary = is_binary(problem)
        if binary
            @assert q == 2 "q should be 2 for binary problems"
        else
            @assert q > 2 "q should be greater than 2 for multi-state problems"
        end
        learning_rate = T(optimizer_type.learning_rate)

        return new{P, O, T}(
            problem, num_trials, num_steps, betas, binary, optimizer_type, learning_rate, T(gamma_grad), manual_grad, T(h_factor), q, seed, device, T
        )
    end
end

function initialize(solver::Solver)
    # Initialize the solver
    # Set the random seed for reproducibility
    Random.seed!(solver.seed)

    if solver.binary
        # Initialize h and p for binary problems
        h = solver.h_factor .* randn(solver.dtype, solver.num_trials, solver.problem.node_num)
    else
        # Initialize h and p for multi-state problems
        h = solver.h_factor .* randn(solver.dtype, solver.num_trials, solver.problem.node_num, solver.q)
    end
    return h
end

function free_energy(solver::Solver, h, step)
    p = solver.binary ? sigmoid.(h) : softmax(h, dims=3)
    # Calculate energy term (expectation value)
    energy = energy_term(solver.problem, p)

    # Calculate entropy term scaled by beta
    entropy = entropy_term(solver.problem, p) ./ solver.betas[step]

    # Free energy = energy - T*entropy (where T = 1/beta)
    return sum(energy .- entropy)
end

function fem_iterate(solver::Solver)
    # Initialize the solver with local fields.
    h = initialize(solver)
    optimizer = get_optimizer(solver.optimizer_type)
    state = Flux.setup(optimizer, h)

    grad = solver.manual_grad ? similar(h) : nothing

    for step in 1:length(solver.betas)
        # Calculate probabilities once per step
        if solver.binary
            p = sigmoid.(h)
        else
            p = softmax(h, dims=3)
        end

        if solver.manual_grad
            grad .= energy_term_grad(solver.problem, p) .+ entropy_term_grad(solver.problem, p) ./ solver.betas[step]
        else
            # Use Zygote for automatic differentiation
            grad = Zygote.gradient(h -> free_energy(solver, h, step), h)[1] 
        end

        Flux.update!(state, h, solver.gamma_grad .* grad)
    end

    # Return final probabilities
    return solver.binary ? sigmoid.(h) : softmax(h, dims=3)
end
