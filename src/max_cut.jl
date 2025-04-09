struct MaxCut <: CombinatorialProblem
    node_num::Int
    edge_num::Int
    coupling::AbstractMatrix
    discretization::Bool
    _grad_normalize_factor::AbstractVector
    dtype::DataType

    function MaxCut(node_num::Int, edge_num::Int, coupling_matrix::AbstractMatrix; discretization::Bool = true, dtype::DataType = Float32)
        @assert node_num == size(coupling_matrix, 1) == size(coupling_matrix, 2)
        # This is the gradient normalization factor cᵢ.
        _grad_normalize_factor = vec(1 ./ sum(abs.(coupling_matrix), dims=2))

        return new(node_num, edge_num, dtype.(coupling_matrix), discretization, dtype.(_grad_normalize_factor), dtype)
    end
end

problem_trait(::Type{MaxCut}) = BinaryProblem()

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
    E = zeros(problem.dtype, batchsize)
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