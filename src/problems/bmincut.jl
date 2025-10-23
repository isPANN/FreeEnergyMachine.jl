struct bMinCut{T<:AbstractFloat, M<:AbstractMatrix{T}} <: CombinatorialProblem
    node_num::Int
    coupling::M
    q::Int
    λ::T

    function bMinCut(
        coupling_matrix::AbstractMatrix, q::Int;
        λ::Real = 5.0,
        device::Union{String, AbstractDevice} = "cpu"
    )
        @assert size(coupling_matrix, 1) == size(coupling_matrix, 2)
        T = eltype(coupling_matrix)
        node_num = size(coupling_matrix, 1)

        W2 = sum(coupling_matrix.^2)
        imbalance_weight = T(λ * W2 / (node_num^2))

        dev = device isa String ? select_device(device) : device
        coupling_device = to_device(dev, T.(coupling_matrix))
        M = typeof(coupling_device)
        
        return new{T, M}(node_num, coupling_device, q, imbalance_weight)
    end
end

is_binary(prob::bMinCut) = prob.q == 2


@inline function _energy_term_first_term(prob::bMinCut{T}, p) where T
    batch_size, N, q = size(p)
    W = prob.coupling
    
    # First term: sum_{(i,j)∈E} W_ij * sum_σ P_i(σ) * [1 - P_j(σ)]
    # Expand: sum_{i,j} W_ij * sum_σ [P_i(σ) - P_i(σ)P_j(σ)]
    #       = sum_{i,j} W_ij * [1 - P_i^T P_j]
    
    # For undirected graph, count each edge once (upper triangular)
    # Get upper triangular part (works efficiently for sparse matrices)
    W_upper = triu(W, 1)
    
    # Compute P_i^T P_j for all pairs: [B, N, N]
    # For each batch b, we compute p[b,:,:] * p[b,:,:]^T
    # Result: P_prod[b,i,j] = sum_k p[b,i,k] * p[b,j,k]
    
    # Use map to avoid mutation
    edge_term = map(1:batch_size) do b
        # p[b,:,:] is [N, q]
        p_b = p[b, :, :]  # [N, q]
        
        # Compute P_i^T P_j = p_b * p_b^T  -> [N, N]
        P_prod = p_b * transpose(p_b)  # [N, N]
        
        # Energy: sum over edges: W_upper_ij * (1 - P_prod_ij)
        sum(W_upper .* (one(T) .- P_prod))
    end
    
    return edge_term
end

function energy_term(prob::bMinCut{T}, p) where T
    @assert ndims(p) == 3 "p must be (B,N,q)"
    batch_size, N, q = size(p)
    @assert N == prob.node_num && q == prob.q
    
    E1 = _energy_term_first_term(prob, p)
    
    if prob.λ != 0
        # Second term: λ * (||P||_F^2 - sum_i ||p_i||^2)
        P_squared = p .^ 2
        frobenius_term = vec(sum(P_squared, dims=[2, 3]))  # [B]
        row_norm_term = vec(sum(sum(P_squared, dims=3), dims=2))  # [B]
        E2 = prob.λ .* (frobenius_term .- row_norm_term)
        return E1 .+ E2
    else
        return E1
    end
end


function infer(prob::bMinCut{T}, p) where T
    # calculate the real cut value of each configuration
    config = one_hot_argmax(p)  # (batchsize, N, q)
    
    # Handle both batched and unbatched input
    if ndims(config) == 2
        config = reshape(config, 1, size(config)...)
    end
    
    batch_size, N, q = size(config)
    W = prob.coupling
    
    # Use upper triangular to avoid double counting (efficient for sparse)
    W_upper = triu(W, 1)
    
    # First term: sum_{(i,j)∈E} W_ij [1 - δ(σ_i, σ_j)]
    # δ(σ_i, σ_j) = config_i^T * config_j (inner product of one-hot vectors)
    
    # Use map to avoid mutation (Zygote-compatible)
    E = map(1:batch_size) do b
        config_b = config[b, :, :]  # [N, q]
        
        # Compute δ(σ_i, σ_j) = config_b * config_b^T -> [N, N]
        delta_ij = config_b * transpose(config_b)
        
        # First term: W_ij * [1 - δ(σ_i, σ_j)] for edges
        edge_term = sum(W_upper .* (one(T) .- delta_ij))
        
        # Second term: λ * sum_{i,j≠i} δ(σ_i, σ_j)
        # sum_{i,j≠i} δ(σ_i, σ_j) = sum(delta_ij) - N (exclude diagonal)
        penalty_term = prob.λ * (sum(delta_ij) - T(N))
        
        edge_term + penalty_term
    end
    
    return batch_size == 1 ? E[1] : E
end