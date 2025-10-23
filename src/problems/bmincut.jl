struct bMinCut{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: CombinatorialProblem
    node_num::Int
    coupling::M
    deg::V
    q::Int
    λ::T

    function bMinCut(
        coupling_matrix::AbstractMatrix, q::Int;
        λ::Real = 5.0,
        device::Union{String, AbstractDevice} = "cpu"
    )
        @assert size(coupling_matrix, 1) == size(coupling_matrix, 2)
        node_num = size(coupling_matrix, 1)
        deg  = vec(sum(coupling_matrix, dims=2))

        W2 = sum(coupling_matrix.^2)
        imbalance_weight = λ * W2 / (node_num^2)

        T = eltype(coupling_matrix)
        
        # Convert device if needed
        dev = device isa String ? select_device(device) : device
        
        # Move to device
        coupling_device = to_device(dev, T.(coupling_matrix))
        deg_device = to_device(dev, T.(deg))
        M = typeof(coupling_device)
        V = typeof(deg_device)

        return new{T, M, V}(node_num, coupling_device, deg_device, q, T(imbalance_weight))
    end
end

is_binary(::bMinCut) = false

@inline function _energy_term_first_term(p, W, X)
    B, N, q = size(p)
    T = promote_type(eltype(W), eltype(p))
    
    d = W * ones(T, N)                  # (N,)
    # ---- First term: ∑_{σ} ∑_{(i,j)} W_ij P_i(σ)[1 - P_j(σ)]
    # dᵀ v - vᵀ W v 
    t_deg  = X * d                      # (Bq,)
    XW     = X * W                      # (Bq,N)
    quad   = vec(sum(X .* XW, dims=2))  # (Bq,)
    e1_rows = T(2) .* (t_deg .- quad)   # (Bq,)
    vec(sum(reshape(e1_rows, B, q), dims=2))  # (B,)
end

function energy_term(prob::bMinCut, p)
    @assert ndims(p) == 3 "p must be (B,N,q)"
    B, N, q = size(p)
    @assert N == prob.node_num && q == prob.q

    W = prob.coupling
    X = reshape(p, B*q, N)              # (Bq, N)
    T = promote_type(eltype(W), eltype(p))

    E1 = _energy_term_first_term(p, W, X)
    # ---- Second term: λ * ∑_{i,j,σ}[P_iP_j - P_i^2] = λ * ∑_{σ}[ (∑ v)^2 - N∑ v^2 ]
    if prob.λ != 0
        S   = sum(X,  dims=2)                   # (Bq,1)
        L2  = sum(X.^2, dims=2)                 # (Bq,1)
        e2_rows = vec(S.^2 .- T(N) .* L2)       # (Bq,)
        E2 = vec(sum(reshape(e2_rows, B, q), dims=2))
        return E1 .+ prob.λ .* E2
    else
        return E1
    end
end


function infer(problem::bMinCut, p)
    # calculate the real cut value of each configuration
    config = one_hot_argmax(p)  # (batchsize, N, q)
    batchsize, N = size(p)
    T = eltype(problem.coupling)
    X = reshape(p, batchsize*problem.q, N)
    _energy_term_first_term(T.(config), problem.coupling, X) .* T(0.5)
end