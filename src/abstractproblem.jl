abstract type CombinatorialProblem end
abstract type BinaryProblem <: CombinatorialProblem end
abstract type MultiStateProblem <: CombinatorialProblem end

function infer(::CombinatorialProblem, p)
    throw(MethodError(infer, (typeof(p),)))
end

function energy_term_grad(::CombinatorialProblem, p)
    throw(MethodError(energy_term_grad, (typeof(p),)))
end

function energy_term(::CombinatorialProblem, p)
    throw(MethodError(energy_term, (typeof(p),)))
end

function entropy_term(::BinaryProblem, p::AbstractMatrix)
    # p: (batch_size, node_num)
    # S_MF = - sum_i p_i * log(p_i) - (1 - p_i) * log(1 - p_i)
    return - sum(p .* log.(p) .+ (1 .- p) .* log.(1 .- p), dims=2)[:]
end

function entropy_term_grad(::BinaryProblem, p::AbstractMatrix)
    # p: (batch_size, node_num)
    return p .* (1 .- p) .* (log.(p) .- log.(1 .- p))
end