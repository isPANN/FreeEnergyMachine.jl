struct MaxCut{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: CombinatorialProblem
    node_num::Int
    coupling::M
    discretization::Bool
    _grad_normalize_factor::V
    q::Int  # MaxCut is always binary (q=2)

    function MaxCut(
        coupling_matrix::AbstractMatrix; 
        discretization::Bool = true,
        device::Union{String, AbstractDevice} = "cpu"
    )
        @assert size(coupling_matrix, 1) == size(coupling_matrix, 2)
        node_num = size(coupling_matrix, 1)
        
        T = eltype(coupling_matrix)
        
        # Convert device if needed
        dev = device isa String ? select_device(device) : device
        
        # This is the gradient normalization factor cᵢ.
        _grad_normalize_factor = vec(1 ./ sum(abs.(coupling_matrix), dims=2))
        
        # Move to device
        coupling_device = to_device(dev, T.(coupling_matrix))
        grad_norm_device = to_device(dev, T.(_grad_normalize_factor))
        
        M = typeof(coupling_device)
        V = typeof(grad_norm_device)
        
        return new{T, M, V}(node_num, coupling_device, discretization, grad_norm_device, 2)
    end
end

is_binary(prob::MaxCut) = prob.q == 2

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
    config = round.(p)  # (batchsize, N)
    T = eltype(p)
    
    # Move coupling to CPU to avoid GPU scalar indexing
    # W_cpu = Array(problem.coupling)
    
    # # Initialize energy on same device as config
    # E = similar(config, T, batchsize)
    # fill!(E, zero(T))
    
    # # E = sum_((i,j) in cal(E)) W_(i,j) (1-delta(p_i, p_j))
    # # Iterate over node pairs, but vectorize over batches
    # for i in 1:N
    #     for j in i+1:N
    #         Wij = W_cpu[i, j]  # Scalar indexing on CPU array
    #         if Wij ≠ 0
    #             # Vectorized operation over all batches (GPU-friendly)
    #             E .+= Wij .* (config[:, i] .!= config[:, j])
    #         end
    #     end
    # end
    - energy_term(problem, config) ./ T(2)
end