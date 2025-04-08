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
    # Compute the energy term of the MaxCut problem
    # U_mf = - sum 2 * W_ij * (1 - p_i) * p_i
    # p: (batch_size, node_num)
    return -2 .* sum(((p .* (1 .- p)) * (problem.coupling / 2) ), dims=2)[:]
end

function infer(problem::MaxCut, p)
    config = round.(p) 
    return sum(((p .* (1 .- p)) * (problem.coupling) ), dims=2)[:]
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