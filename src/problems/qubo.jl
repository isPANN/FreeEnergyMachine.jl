struct QUBO <: CombinatorialProblem
    node_num::Int
    # edge_num::Int
    coupling::AbstractMatrix
    discretization::Bool
    _grad_normalize_factor::AbstractVector
    dtype::DataType

    function QUBO(coupling_matrix::AbstractMatrix; discretization::Bool = true, dtype::DataType = Float32)
        @assert size(coupling_matrix, 1) == size(coupling_matrix, 2)
        node_num = size(coupling_matrix, 1)
        
        # This is the gradient normalization factor cᵢ.
        _grad_normalize_factor = vec(1 ./ sum(abs.(coupling_matrix), dims=2))

        return new(node_num, dtype.(coupling_matrix), discretization, dtype.(_grad_normalize_factor), dtype)
    end
end