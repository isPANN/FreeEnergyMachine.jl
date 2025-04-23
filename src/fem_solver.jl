struct Solver{RT<:AbstractRule, T<:AbstractFloat}
    num_trials::Int
    num_steps::Int
    betas::Vector{T}
    optimizer::RT
    gamma_grad::T
    h_factor::T
    flavor::Int

    function Solver(
        num_trials::Int,
        num_steps::Int,
        betas;
        gamma_grad::Real = 1.0,
        optimizer::RT = Flux.Adam(0.01,(0.9,0.999)),
        h_factor::Real = 0.01,
        flavor::Int = 2,
    ) where {RT <:AbstractRule}
        T = typeof(betas[1])
        return new{RT, T}(
             num_trials, num_steps, betas,  optimizer, T(gamma_grad), T(h_factor), flavor
        )
    end
end

function initialize(problem,solver::Solver{O, T}) where {O, T}
    # Initialize the solver

    if solver.flavor == 2
        # Initialize h and p for binary problems
        h = solver.h_factor .* randn(T, solver.num_trials, problem.node_num)
    else
        # Initialize h and p for multi-state problems
        h = solver.h_factor .* randn(T, solver.num_trials, problem.node_num, solver.flavor)
    end
    return h
end

function free_energy(problem,solver::Solver, h, step)
    p = solver.flavor == 2 ? sigmoid.(h) : softmax(h, dims=3)
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
    # optimizer = get_optimizer(solver.optimizer_type)
    state = Flux.setup(solver.optimizer, h)

    grad =  similar(h)

    for step in 1:length(solver.betas)
        # Calculate probabilities once per step
        if solver.flavor == 2
            p = sigmoid.(h)
        else
            p = softmax(h, dims=3)
        end

        grad .= energy_term_grad(problem, p) .+ entropy_term_grad(problem, p) ./ solver.betas[step]
        # grad .= (1+1/solver.betas[step]).* entropy_term_grad(problem, p)
        # Use Zygote for automatic differentiation
        # grad2 = Zygote.gradient(h -> free_energy(problem,solver, h, step), h)[1] 
        # gval = zero(h)
        # _, fval = Enzyme.autodiff(ReverseWithPrimal, Const(h -> free_energy(problem,solver, h, step)), Active, Duplicated(h, gval))
        # @show grad
        # @show grad2
        # @show fval
        # @show gval
        # @assert false

        Flux.update!(state, h, solver.gamma_grad .* grad)
    end

    # Return final probabilities
    return solver.flavor == 2 ? sigmoid.(h) : softmax(h, dims=3)
end
