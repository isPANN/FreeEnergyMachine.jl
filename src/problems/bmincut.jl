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

        W_upper_cpu = triu(T.(coupling_matrix), 1)

        dev = device isa String ? select_device(device) : device
        coupling_device = to_device(dev, W_upper_cpu)
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
    W_upper = prob.coupling

    # Convert to dense array first (避免稀疏矩阵在 GPU 上的标量索引问题)
    W_dense = Array(W_upper)
    
    # Match device & eltype with p
    Wd = (p isa CuArray) ? cu(W_dense) : W_dense
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

function one_hot_argmax(p::Array{T,3}) where {T<:Real}
    @assert ndims(p) == 3 "p must be (batch, node, q)"
    B, N, Q = size(p)
    
    # Reshape to 2D, find argmax, then one-hot encode
    p_flat = reshape(p, B * N, Q)
    idx_flat = mapslices(argmax, p_flat; dims=2) |> vec  # (B*N,)
    
    # Create one-hot matrix (use same type as input)
    s_flat = zeros(T, B * N, Q)
    for i in 1:(B * N)
        s_flat[i, idx_flat[i]] = one(T)
    end
    
    return reshape(s_flat, B, N, Q)
end

function one_hot_argmax(p::CuArray{T,3}) where {T<:Real}
    # [B,N,1]
    m = maximum(p; dims=3)
    # [B,N,Q] CuArray; ties -> multi-1s（若需严格 argmax，可再打破并列）
    return T.(p .== m)  # 转换为浮点类型以保持一致性
end

function infer(prob::bMinCut{T}, p) where T
    config = one_hot_argmax(p)
    
    if ndims(config) == 2
        config = reshape(config, 1, size(config)...)
    end
    
    batch_size, N, q = size(config)
    W_upper = prob.coupling
    
    # Convert to dense array first (避免稀疏矩阵在 GPU 上的标量索引问题)
    W_dense = Array(W_upper)
    
    # Match device with config
    W_upper = (config isa CuArray) ? cu(W_dense) : W_dense
    
    # Vectorized computation (no scalar indexing)
    # config: [B, N, q]
    # Compute delta_ij for all batches: [N, q, B] * [q, N, B] -> [N, N, B]
    config_NqB = permutedims(config, (2, 3, 1))  # [N, q, B]
    delta_NNB = batched_mul(config_NqB, permutedims(config_NqB, (2, 1, 3)))  # [N, N, B]
    
    # Edge term: sum over (i,j) for each batch
    # W_upper: [N, N], delta_NNB: [N, N, B]
    edge_term = vec(sum(reshape(W_upper, N, N, 1) .* (one(T) .- delta_NNB); dims=(1, 2)))  # [B]
    
    # Penalty term: λ * (sum(delta_ij) - N) for each batch
    penalty_term = prob.λ .* (vec(sum(delta_NNB; dims=(1, 2))) .- T(N))  # [B]
    
    E = edge_term .+ penalty_term  # [B]
    
    return batch_size == 1 ? E[1] : E
end