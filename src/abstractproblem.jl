abstract type CombinatorialProblem end

function is_binary(problem::CombinatorialProblem)
    throw(MethodError(is_binary, (typeof(problem),)))
end

function load_problem(prob::Type{<:CombinatorialProblem}, args...)
    throw(MethodError(load_problem, (prob, args...)))
end

function infer(::CombinatorialProblem, p)
    throw(MethodError(infer, (typeof(p),)))
end

function energy_term_grad(::CombinatorialProblem, p)
    throw(MethodError(energy_term_grad, (typeof(p),)))
end

function energy_term(::CombinatorialProblem, p)
    throw(MethodError(energy_term, (typeof(p),)))
end

function entropy_term(problem::CombinatorialProblem, p)
    # q represents the number of states of each spin.
    if is_binary(problem)
        return _entropy_binary(p)
    else
        return _entropy_q(p)
    end
end

function entropy_term_grad(problem::CombinatorialProblem, p)
    # q represents the number of states of each spin.
    if is_binary(problem)
        g = similar(p)
        _entropy_grad_binary!(g, p)
        return g
    else
        # return entropy_grad_q(p)
        throw(ArgumentError("To Be Implemented"))
    end
end

@inline function _h(x::T) where {T<:Real}
    T0 = eps(T)
    x = clamp(x, T0, one(T))
    return -x * log(x)
end

function _entropy_q(p::AbstractArray)
    # p: (batch_size, node_num, q)
    # S_MF = - sum_i sum_σᵢ∈{1,2,...,q} p_i(σᵢ) log(p_i(σᵢ))
    vec(sum(_h.(p), dims=(2,3)))
end

# function entropy_grad_q(p)
#     logp = log.(p)
#     sumlogp = sum(p .* logp, dims=3)
#     return -p .* (logp .- sumlogp)
# end

@inline function _hb(x::T) where {T<:Real}
    x  = clamp(x, eps(T), one(T) - eps(T))
    y  = one(T) - x
    - (x * log(x) + y * log1p(-x))      
end

function _entropy_binary(p::AbstractMatrix)
    # p: (batch_size, node_num)
    # S_MF = - sum_i [p_i * log(p_i) + (1 - p_i) * log(1 - p_i)]
    vec(sum(_hb.(p), dims=2))
end

@inline function _hb_g(x::T) where {T<:Real}
    x = clamp(x, eps(T), one(T) - eps(T))
    y = one(T) - x
    x * y * (log(x) - log1p(-x))
end

function _entropy_grad_binary!(g::AbstractMatrix, p::AbstractMatrix)
    # p: (batch_size, node_num)
    # return p .* (1 .- p) .* (log.(p) .- log.(1 .- p))
    @assert size(g) == size(p)
    @. g = _hb_g(p)
    return g
end