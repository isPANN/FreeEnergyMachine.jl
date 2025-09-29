struct SolverConfig{A<:AnnealingStrategy, O<:OptimizerType}
    betamin::Float64
    betamax::Float64
    annealing::A
    optimizer::O
    gamma_grad::Float64
    manual_grad::Bool
    h_factor::Float64
    seed::Int
    device::String
    
    function SolverConfig(;
        betamin::Real = 0.01,
        betamax::Real = 0.5,
        annealing::A = ExponentialAnnealing(),
        optimizer::O = AdamOpt(),
        gamma_grad::Real = 1.0,
        manual_grad::Bool = false,
        h_factor::Real = 0.01,
        seed::Int = 1,
        device::String = "cpu"
    ) where {A<:AnnealingStrategy, O<:OptimizerType}
        new{A, O}(betamin, betamax, annealing, optimizer, gamma_grad, manual_grad, h_factor, seed, device)
    end
end

struct Solver{P<:CombinatorialProblem, T<:AbstractFloat}
    problem::P
    num_trials::Int
    num_steps::Int
    betas::Vector{T}
    inv_betas::Vector{T}
    binary::Bool
    q::Int
    config::SolverConfig

    function Solver(
        problem::P,
        num_trials::Int,
        num_steps::Int,
        q::Int = 2;
        config::SolverConfig = SolverConfig()
    ) where {P<:CombinatorialProblem}

        T = eltype(problem.coupling)  # Get the type from the problem's coupling matrix
        betas = T.(get_betas(config.annealing, num_steps, T(config.betamin), T(config.betamax)))
        inv_betas = one(T) ./ betas

        binary = is_binary(problem)
        if binary
            @assert q == 2 "q should be 2 for binary problems"
        else
            @assert q > 2 "q should be greater than 2 for multi-state problems"
        end

        return new{P, T}(
            problem, num_trials, num_steps, betas, inv_betas, binary, q, config
        )
    end
end

function initialize(solver::Solver)
    # Initialize the solver
    # Set the random seed for reproducibility
    Random.seed!(solver.config.seed)

    T = eltype(solver.betas)  # Get the type from the betas vector
    if solver.binary
        # Initialize h and p for binary problems
        h = solver.config.h_factor .* randn(T, solver.num_trials, solver.problem.node_num)
    else
        # Initialize h and p for multi-state problems
        h = solver.config.h_factor .* randn(T, solver.num_trials, solver.problem.node_num, solver.q)
    end
    return h
end

function free_energy(solver::Solver, h, step)
    p = solver.binary ? sigmoid.(h) : softmax(h, dims=3)
    # Calculate energy term (expectation value)
    energy = energy_term(solver.problem, p)

    # Calculate entropy term scaled by beta
    entropy = entropy_term(solver.problem, p) .* solver.inv_betas[step]

    # Free energy = energy - T*entropy (where T = 1/beta)
    return sum(energy .- entropy)
end

function fem_iterate(solver::Solver)
    # Initialize the solver with local fields.
    h = initialize(solver)
    optimizer = get_optimizer(solver.config.optimizer)
    state = Flux.setup(optimizer, h)

    grad = solver.config.manual_grad ? similar(h) : nothing
    pbuf = solver.config.manual_grad && solver.binary ? similar(h) : nothing

    for step in 1:length(solver.betas)
        if solver.config.manual_grad
            # Calculate probabilities once per step only when needed
            if solver.binary
                p = pbuf
                @. p = 1 / (1 + exp(-h))
            else
                p = softmax(h, dims=3)
            end
            grad .= energy_term_grad(solver.problem, p) .+ entropy_term_grad(solver.problem, p) .* solver.inv_betas[step]
        else
            # Use Zygote for automatic differentiation
            grad = Zygote.gradient(h -> free_energy(solver, h, step), h)[1] 
        end

        Flux.update!(state, h, solver.config.gamma_grad .* grad)
    end

    # Return final probabilities
    return solver.binary ? sigmoid.(h) : softmax(h, dims=3)
end
