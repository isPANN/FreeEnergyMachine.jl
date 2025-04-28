struct MaxCut{T} <: BinaryProblem
    node_num::Int
    edge_num::Int
    coupling::Matrix{T}
    discretization::Bool
    _grad_normalize_factor::Vector{T}

    function MaxCut(node_num::Int, edge_num::Int, coupling_matrix::Matrix{T}; discretization::Bool = true) where T
        @assert node_num == size(coupling_matrix, 1) == size(coupling_matrix, 2)
        # This is the gradient normalization factor cᵢ.
        _grad_normalize_factor = vec(1 ./ sum(abs.(coupling_matrix), dims=2))

        return new{T}(node_num, edge_num, coupling_matrix, discretization, _grad_normalize_factor)
    end
end

# function energy_term_grad(problem::MaxCut, p)
#     # Compute the gradient of the MaxCut problem
#     # p: (batch_size, node_num)
#     p_prime = problem.discretization ? round.(p) : p  # each element is eⱼ(+1)
#     _inside_bracket_term = (2 .* p_prime .- 1) * problem.coupling' .* problem._grad_normalize_factor' # (batch_size, node_num)
#     return _inside_bracket_term .* p .* (1 .- p) # (batch_size, node_num)
# end

function energy_term(problem::MaxCut{T}, p::AbstractMatrix{T}) where T
    U = zeros(T, size(p, 1))
    energy_term!(U, problem, p)
    return U
end

function energy_term!(U::AbstractVector{T}, problem::MaxCut{T}, p::AbstractMatrix{T}) where T
    W = problem.coupling  # (N, N)
    # term1 = sum over i,j of W_ij * (P_i + P_j) = 2 * sum(W * p) over batch
    overlap!(U, p, W, fill(T(-2), size(p, 1), size(p, 2)))  # shape: (batch_size, 1)

    # term2 = 2 * sum(W_ij * P_i * P_j)
    # (p * W) gives (batch_size, N)
    # then element-wise multiply with p and sum over nodes
    overlap!(U, p, W, 2 .* p)  # shape: (batch_size, 1)
    return U  # return shape: (batch_size,)
end

function overlap!(res::AbstractVector{T}, p1::AbstractMatrix{T}, J::AbstractMatrix{T}, p2::AbstractMatrix{T}) where T
    @assert size(p1, 1) == size(p2, 1) == size(res, 1)
    @assert size(J) == (size(p1, 2), size(p2, 2))
    for ib in axes(p1, 1), j in axes(p2, 2), i in axes(p1, 2)
        ri = p1[ib, i] * J[i, j] * p2[ib, j]
        res[ib] += ri
    end
end

function infer(problem::MaxCut{T}, p) where T
    # calculate the real cut value of each configuration
    config = round.(p)
    batchsize, N = size(p)
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
    val, pos = findmax(E)
    return config[pos, :], val
end

function load_matrix(path::String; zero_based::Bool = false, symmetric::Bool = true, dtype::Type = Float32)
    @info "Use dtype: $dtype"
    open(path, "r") do io
        first_line = readline(io)
        num_nodes, num_edges = parse.(Int, split(first_line))
        A = spzeros(dtype, num_nodes, num_nodes)
        node1 = Int[]; node2 = Int[]; weights = dtype[]

        for line in eachline(io)
            parts = split(line)
            if length(parts) ≥ 2
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                w = length(parts) ≥ 3 ? parse(dtype, parts[3]) : 1
                if zero_based
                    u += 1; v += 1
                end
                push!(node1, u)
                push!(node2, v)
                push!(weights, w)
            end
        end
        # sort the edges by node1 and node2
        perm = sortperm(node1 * num_nodes .+ node2)
        node1 = node1[perm]
        node2 = node2[perm]
        weights = weights[perm]

        for i in axes(node1, 1)
            A[node1[i], node2[i]] = weights[i]
            if symmetric
                A[node2[i], node1[i]] = weights[i]
            end
        end
        return A, num_nodes, num_edges
    end
end