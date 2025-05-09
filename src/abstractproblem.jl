abstract type CombinatorialProblem end

# ------------------------------
abstract type ProblemTrait end

struct BinaryProblem <: ProblemTrait end
struct MultiStateProblem <: ProblemTrait end

problem_trait(::Type{<:CombinatorialProblem}) = MultiStateProblem()
is_binary(problem::CombinatorialProblem) = isa(problem_trait(typeof(problem)), BinaryProblem)

# ------------------------------
function infer(::CombinatorialProblem, p)
    throw(MethodError(infer, (typeof(p),)))
end

function energy_term_grad(::CombinatorialProblem, p)
    throw(MethodError(energy_term_grad, (typeof(p),)))
end

function energy_term(::CombinatorialProblem, p)
    throw(MethodError(energy_term, (typeof(p),)))
end

function entropy_term(problem::CombinatorialProblem, p::AbstractMatrix)
    # q represents the number of states of each spin.
    if is_binary(problem)
        return _entropy_binary(p)
    else
        # return entropy_q(p)
        throw(ArgumentError("To Be Implemented"))
    end
end

function entropy_term_grad(problem::CombinatorialProblem, p::AbstractMatrix)
    # q represents the number of states of each spin.
    if is_binary(problem)
        return _entropy_grad_binary(p)
    else
        # return entropy_grad_q(p)
        throw(ArgumentError("To Be Implemented"))
    end
end

# function entropy_q(p)
#     return -sum(p .* log.(p), dims=3) |> x -> sum(x, dims=2) |> vec
# end

# function entropy_grad_q(p)
#     logp = log.(p)
#     sumlogp = sum(p .* logp, dims=3)
#     return -p .* (logp .- sumlogp)
# end

function _entropy_binary(p::AbstractMatrix)
    # p: (batch_size, node_num)
    # S_MF = - sum_i p_i * log(p_i) - (1 - p_i) * log(1 - p_i)
    return - sum(p .* log.(p) .+ (1 .- p) .* log.(1 .- p), dims=2)[:]
end

function _entropy_grad_binary(p::AbstractMatrix)
    # p: (batch_size, node_num)
    return p .* (1 .- p) .* (log.(p) .- log.(1 .- p))
end