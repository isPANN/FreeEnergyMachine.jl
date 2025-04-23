struct Solver{O<:OptimizerType, T<:AbstractFloat}
    num_trials::Int
    num_steps::Int
    betas::Vector{T}
    binary::Bool
    optimizer_type::O
    learning_rate::T
    gamma_grad::T
    h_factor::T
    flavor::Int

    function Solver(
        num_trials::Int,
        num_steps::Int;
        betamin::Real = 0.01,
        betamax::Real = 0.5,
        gamma_grad::Real = 1.0,
        annealing::A = ExponentialAnnealing(),
        optimizer_type::O = AdamOpt(),
        h_factor::Real = 0.01,
        flavor::Int = 2,
    ) where { A<:AnnealingStrategy, O<:OptimizerType}

        T = typeof(betamin)
        betas = T.(get_betas(annealing, num_steps, T(betamin), T(betamax)))
        learning_rate = T(optimizer_type.learning_rate)

        return new{O, T}(
             num_trials, num_steps, betas, flavor == 2, optimizer_type, learning_rate, T(gamma_grad), T(h_factor), flavor
        )
    end
end

function initialize(problem,solver::Solver{O, T}) where {O, T}
    # Initialize the solver

    if solver.binary
        # Initialize h and p for binary problems
        h = solver.h_factor .* randn(T, solver.num_trials, problem.node_num)
    else
        # Initialize h and p for multi-state problems
        h = solver.h_factor .* randn(T, solver.num_trials, problem.node_num, solver.flavor)
    end
    return h
end

function free_energy(problem,solver::Solver, h, step)
    p = solver.binary ? sigmoid.(h) : softmax(h, dims=3)
    # Calculate energy term (expectation value)
    energy = energy_term(problem, p)

    # Calculate entropy term scaled by beta
    entropy = entropy_term(problem, p) ./ solver.betas[step]

    # Free energy = energy - T*entropy (where T = 1/beta)
    return sum(energy .- entropy)
end

function fem_iterate(problem,solver::Solver)
    # Initialize the solver with local fields.
    h = initialize(problem,solver)
    optimizer = get_optimizer(solver.optimizer_type)
    state = Flux.setup(optimizer, h)

    grad =  similar(h)

    for step in 1:length(solver.betas)
        # Calculate probabilities once per step
        if solver.binary
            p = sigmoid.(h)
        else
            p = softmax(h, dims=3)
        end

        # grad .= energy_term_grad(problem, p) .+ entropy_term_grad(problem, p) ./ solver.betas[step]
        grad .= (1+1/solver.betas[step]).* entropy_term_grad(problem, p)
            # Use Zygote for automatic differentiation
        grad2 = Zygote.gradient(h -> free_energy(problem,solver, h, step), h)[1] 
        gval = zero(h)
        _, fval = Enzyme.autodiff(ReverseWithPrimal, Const(h -> free_energy(problem,solver, h, step)), Active, Duplicated(h, gval))
        @show grad
        @show grad2
        @show fval
        @show gval
        @assert false

        Flux.update!(state, h, solver.gamma_grad .* grad)
    end

    # Return final probabilities
    return solver.binary ? sigmoid.(h) : softmax(h, dims=3)
end
