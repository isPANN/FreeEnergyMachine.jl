struct FEMQEC{T} <: BinaryProblem
	node_num::Int
	edge_num::Int
	coupling::Matrix{T}
	discretization::Bool
	_grad_normalize_factor::Vector{T}

end

function energy_term(problem::FEMQEC, p)

end

function _even_probability(pvec::Vector{T}) where T
	peven = one(T)
	podd = zero(T)
	for p in pvec
		peven, podd = (one(T) - p) * peven + p * podd, (one(T) - p) * podd + p * peven
	end
	return peven,podd
end
