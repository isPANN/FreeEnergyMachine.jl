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

function _energy_term_first_term(prob::bMinCut{T}, p) where {T}
    # p: [B, N, q]
    @assert ndims(p) == 3 "p must be a B×N×q tensor"
    B, N, q = size(p)

    # Count each undirected edge once
    W_upper = triu(prob.coupling, 1)

    # Match device & eltype with p
    Wd = (p isa CuArray) ? cu(W_upper) : W_upper
    Wd = T.(Wd)  # ensure same element type as p

    # Build per-batch Pi Pj^T: [N,q,B] * [q,N,B] -> [N,N,B]
    p_NqB = permutedims(p, (2, 3, 1))                     # [N, q, B]
    P_NNB = batched_mul(p_NqB, permutedims(p_NqB, (2,1,3)))  # [N, N, B]

    # E_b = sum_{i,j} W_ij * (1 - P_ijb), sum over (i,j), keep batch dim
    E = sum(reshape(Wd, N, N, 1) .* (one(T) .- P_NNB); dims=(1,2))  # [1,1,B]

    return vec(dropdims(E; dims=(1,2)))  # [B], lives on same device as p
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

function one_hot_argmax(p)
    @assert ndims(p) == 3 "p must be (batch, node, q)"
    B, N, Q = size(p)
    
    # Reshape to 2D, find argmax, then one-hot encode
    p_flat = reshape(p, B * N, Q)
    idx_flat = mapslices(argmax, p_flat; dims=2) |> vec  # (B*N,)
    
    # Create one-hot matrix
    s_flat = falses(B * N, Q)
    for i in 1:(B * N)
        s_flat[i, idx_flat[i]] = true
    end
    
    return reshape(s_flat, B, N, Q)
end

function infer(prob::bMinCut{T}, p) where T
    config = one_hot_argmax(p)
    
    if ndims(config) == 2
        config = reshape(config, 1, size(config)...)
    end
    
    batch_size, N, q = size(config)
    W = prob.coupling
    W_upper = triu(W, 1)
    
    # Pre-allocate results
    E = similar(config, T, batch_size)
    
    # Efficient loop (GPU kernels will fuse operations)
    for b in 1:batch_size
        config_b = @view config[b, :, :]  # [N, q]
        
        # Compute δ(σ_i, σ_j)
        delta_ij = config_b * transpose(config_b)  # [N, N]
        
        # Edge term
        edge_term = sum(W_upper .* (one(T) .- delta_ij))
        
        # Penalty term
        penalty_term = prob.λ * (sum(delta_ij) - T(N))
        
        E[b] = edge_term + penalty_term
    end
    
    return batch_size == 1 ? E[1] : E
end