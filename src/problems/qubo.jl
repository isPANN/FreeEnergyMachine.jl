struct QUBO{T<:AbstractFloat, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: CombinatorialProblem
    node_num::Int
    # edge_num::Int
    coupling::M
    discretization::Bool
    _grad_normalize_factor::V

    function QUBO(
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
        
        return new{T, M, V}(node_num, coupling_device, discretization, grad_norm_device)
    end
end

is_binary(problem::QUBO) = true