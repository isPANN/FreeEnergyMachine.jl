struct QUBO{T<:AbstractFloat} <: CombinatorialProblem
    node_num::Int
    # edge_num::Int
    coupling::AbstractMatrix{T}
    discretization::Bool
    _grad_normalize_factor::AbstractVector{T}

    function QUBO(coupling_matrix::AbstractMatrix; discretization::Bool = true)
        @assert size(coupling_matrix, 1) == size(coupling_matrix, 2)
        node_num = size(coupling_matrix, 1)
        
        # This is the gradient normalization factor cᵢ.
        _grad_normalize_factor = vec(1 ./ sum(abs.(coupling_matrix), dims=2))

        T = eltype(coupling_matrix)
        return new{T}(node_num, T.(coupling_matrix), discretization, T.(_grad_normalize_factor))
    end
end

problem_trait(::Type{<:QUBO}) = BinaryProblem()