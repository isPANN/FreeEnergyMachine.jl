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

function initialize(problem::MaxCut{T}, solver::Solver{O, T}) where {O, T}
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

function fem_iterate(problem::MaxCut{T}, solver::Solver{O, T}) where {O, T}
    # Initialize the solver with local fields.
    h = initialize(problem, solver)
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

        grad .= energy_term_grad(problem, h) .+ entropy_term_grad(problem, p) ./ solver.betas[step]

        Flux.update!(state, h, solver.gamma_grad .* grad)
    end

    # Return final probabilities
    return solver.flavor == 2 ? sigmoid.(h) : softmax(h, dims=3)
end

# function energy_term_grad(problem, h::AbstractMatrix{T}) where T
#     gval = similar(h)
#     function energy_term_grad_wrapper(U, problem, h)
#         p = sigmoid.(h)
#         return sum(energy_term!(U, problem, p))
#     end
#     U = zeros(T, size(h, 1))
#     gU = zero(U)
#     Enzyme.autodiff(Reverse, energy_term_grad_wrapper, Active, Duplicated(U, gU), Const(problem), Duplicated(h, gval))
#     return gval
# end

function energy_term_grad(problem, h::AbstractMatrix{T}) where T
    gval = similar(h)
    function energy_term_grad_wrapper( problem, h)
        p = sigmoid.(h)
        return sum(energy_term(problem, p))
    end
    Enzyme.autodiff(Reverse, energy_term_grad_wrapper, Active, Const(problem), Duplicated(h, gval))
    return gval
end

# function energy_term_grad(problem::MaxCut{T}, h::AbstractMatrix{T}) where T
#     p = sigmoid.(h)
#     # Compute the gradient of the MaxCut problem
#     # p: (batch_size, node_num)
#     p_prime = problem.discretization ? round.(p) : p  # each element is eⱼ(+1)
#     _inside_bracket_term = (2 .* p_prime .- 1) * problem.coupling' .* problem._grad_normalize_factor' # (batch_size, node_num)
#     return _inside_bracket_term .* p .* (1 .- p) # (batch_size, node_num)
# end