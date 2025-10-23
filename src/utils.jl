function load_weighted_graph(path::String; zero_based::Bool = false, symmetric::Bool = true, dtype::Type = Float32)
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

function one_hot_argmax(p)
    @assert ndims(p) == 3 "p must be (batch, node, q)"
    B, N, Q = size(p)

    idx = mapslices(argmax, p; dims=3)
    idx = dropdims(idx; dims=3)  # (B, N)

    s = falses(B, N, Q)
    for b in 1:B, i in 1:N
        s[b, i, idx[b, i]] = true
    end
    return s
end