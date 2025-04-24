struct FEMQEC{T} <: BinaryProblem
	ops::VecPtr{VIT,VIT}
	ops_check::VecPtr{VIT,VIT}
	logp::VecPtr{VT,VIT}
	logp2bit::VecPtr{VIT,VIT}
	bit2logp::VecPtr{VIT,VIT}
	bit2ops::VecPtr{VIT,VIT}
end


# struct SpinGlassSA{VT, VIT, T} 
# 	ops::VecPtr{VIT,VIT}
# 	ops_check::VecPtr{VIT,VIT}
# 	logp::VecPtr{VT,VIT}
# 	logp2bit::VecPtr{VIT,VIT}
# 	bit2logp::VecPtr{VIT,VIT}
# 	betas::Vector{T}
# 	num_trials::Int
# 	partitions::Vector{Vector{Int}}
# end

function SpinGlassSA2FEMQEC(tanner::TannerGraph, em::ErrorModel, pvec::Vector{T}) where T
	coupling = tanner.coupling
	discretization = tanner.discretization
	_grad_normalize_factor = tanner._grad_normalize_factor
	return FEMQEC(tanner.node_num, tanner.edge_num, coupling, discretization, _grad_normalize_factor)
end

function energy_term(sa::SpinGlassSA{VT, VIT, T}, pvec::Vector{T},config::Vector{Mod2}) where {VT, VIT, T}
	E = zero(T)
	for j in 1:length(sa.logp)
		for (pos,logp_val) in getview(sa.logp,j)
		for bit_num in getview(sa.logp2bit,j)
			for i in getview(sa.bit2ops,bit_num)
				E += config[i] * config[j] * sa.coupling[i,j]
			end
		end
	end
	return E
end

function _even_probability(pvec::Vector{T}) where T
	peven = one(T)
	podd = zero(T)
	for p in pvec
		peven, podd = (one(T) - p) * peven + p * podd, (one(T) - p) * podd + p * peven
	end
	return peven,podd
end
