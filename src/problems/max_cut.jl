struct MaxCut{T<:AbstractFloat} <: CombinatorialProblem
    node_num::Int
    # edge_num::Int
    coupling::AbstractMatrix{T}
    discretization::Bool
    _grad_normalize_factor::AbstractVector{T}

    function MaxCut(coupling_matrix::AbstractMatrix; discretization::Bool = true)
        @assert size(coupling_matrix, 1) == size(coupling_matrix, 2)
        node_num = size(coupling_matrix, 1)
        
        # This is the gradient normalization factor cᵢ.
        _grad_normalize_factor = vec(1 ./ sum(abs.(coupling_matrix), dims=2))

        T = eltype(coupling_matrix)
        return new{T}(node_num, T.(coupling_matrix), discretization, T.(_grad_normalize_factor))
    end
end

problem_trait(::Type{<:MaxCut}) = BinaryProblem()

function energy_term_grad(problem::MaxCut, p)
    # Compute the gradient of the MaxCut problem
    # p: (batch_size, node_num)
    p_prime = problem.discretization ? round.(p) : p  # each element is eⱼ(+1)
    _inside_bracket_term = (2 .* p_prime .- 1) * problem.coupling' .* problem._grad_normalize_factor' # (batch_size, node_num)
    return _inside_bracket_term .* p .* (1 .- p) # (batch_size, node_num)
end

function energy_term(problem::MaxCut, p)
    W = problem.coupling  # (N, N)
    # term1 = sum over i,j of W_ij * (P_i + P_j) = 2 * sum(W * p) over batch
    term1 = 2 .* sum(p * W; dims=2)  # shape: (batch_size, 1)

    # term2 = 2 * sum(W_ij * P_i * P_j)
    # (p * W) gives (batch_size, N)
    # then element-wise multiply with p and sum over nodes
    term2 = 2 .* sum(p .* (p * W); dims=2)  # shape: (batch_size, 1)

    U = -(term1 .- term2)  # shape: (batch_size, 1)
    return vec(U)  # return shape: (batch_size,)
end

function infer(problem::MaxCut, p)
    # calculate the real cut value of each configuration
    config = round.(p)
    batchsize, N = size(p)
    T = eltype(problem.coupling)
    E = zeros(T, batchsize)
    # E = sum_((i,j) in cal(E)) W_(i,j) (1-delta(p_i, p_j))
    for i in 1:N, j in i+1:N
        Wij = problem.coupling[i, j]
        if Wij ≠ 0
            for b in 1:batchsize
                E[b] += Wij * (config[b, i] ≠ config[b, j])
            end
        end
    end
    return E
end