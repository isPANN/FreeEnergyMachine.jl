struct FEMQEC{T,IT} <: BinaryProblem{T}
	ops::VecPtr{IT,IT}
	ops_check::VecPtr{IT,IT}
	logp::VecPtr{T,IT}
	logp2bit::VecPtr{IT,IT}
	bit2logp::VecPtr{IT,IT}
	bit2ops::VecPtr{IT,IT}
	config::Vector{Mod2}
end


function generate_femqec(tanner::CSSTannerGraph, ide::IndependentDepolarizingError{T},config::Vector{Mod2}; IT=Int32) where T
	qubit_num = tanner.stgx.nq
    check_number = tanner.stgx.ns + tanner.stgz.ns

	lx,lz = logical_operator(tanner)
	xlogical_qubits = [findall(i->i.x,row) for row in eachrow(lx)]
    lx_num = length(xlogical_qubits)
	zlogical_qubits = [findall(i->i.x,row) for row in eachrow(lz)]
    q2xlogical =  [check_number .+ findall(x-> i ∈ x , xlogical_qubits) for i in 1:qubit_num]
    q2zlogical =  [(lx_num + check_number) .+ findall(x-> i ∈ x , zlogical_qubits) for i in 1:qubit_num]
	q2logical = vcat(q2xlogical,q2zlogical)

	vecvecops = vcat(tanner.stgx.s2q, broadcast.(+,tanner.stgz.s2q,qubit_num),xlogical_qubits, broadcast.(+,zlogical_qubits,qubit_num))
	ops = _vecvec2vecptr(vecvecops, IT,IT)
	ops_check = _vecvec2vecptr(vcat(zlogical_qubits, broadcast.(+,xlogical_qubits,qubit_num)), IT,IT)
	# ops_correct = _vecvec2vecptr(vcat(xlogical_qubits, broadcast.(+,zlogical_qubits,qubit_num)), IT,IT)
	logp = _vecvec2vecptr([[log(one(T)-px-py-pz),log(px),log(pz),log(py)] for (px,py,pz) in zip(ide.px,ide.py,ide.pz)], IT,T)
	logp2bit = _vecvec2vecptr([[i,i+qubit_num] for i in 1:qubit_num], IT,IT)
	bit_vec = [[i] for i in 1:qubit_num]
	bit2logp = _vecvec2vecptr(vcat(bit_vec,bit_vec), IT,IT)

	bit2ops = _vecvec2vecptr(vcat.(vcat(tanner.stgx.q2s,broadcast.(+,tanner.stgz.q2s,tanner.stgx.ns)),q2logical), IT,IT)
	return FEMQEC(ops, ops_check, logp, logp2bit, bit2logp, bit2ops,config)
end

function energy_term(sa::FEMQEC{T,IT}, pvec::AbstractMatrix{T}) where {T,IT}
	batch_size = size(pvec, 1)
	E = zeros(T, batch_size)	
	bit_num = length(sa.bit2logp)

	for b in 1:batch_size
		peven_vec = Matrix{T}(undef,2,bit_num)
		# peven_vec[1,j] stands for the probability of j-th qubit is 0
		# peven_vec[2,j] stands for the probability of j-th qubit is 1

		for i in 1:bit_num
			peven,podd = _even_probability(view(pvec[b,:],getview(sa.bit2ops,i)))
			peven_vec[1,i],peven_vec[2,i] = sa.config[i].x ? (podd,peven) : (peven,podd)
		end

		for j in 1:length(sa.logp)
			view_j = getview(sa.logp2bit,j)
			bit_num_j = length(view_j)
			for (pos,val) in enumerate(getview(sa.logp,j))
				E[b] += prod(i -> peven_vec[1+ readbit(pos-1,i),view_j[i]], 1:bit_num_j) * val
			end
		end
	end
	return E
end

function _even_probability(pvec::AbstractVector{T}) where T
	peven = one(T)
	podd = zero(T)
	for p in pvec
		peven, podd = (one(T) - p) * peven + p * podd, (one(T) - p) * podd + p * peven
	end
	return peven,podd
end

function optimal_energy(em::IndependentDepolarizingError{T}, tanner::CSSTannerGraph,syd) where {T}
	prob,_ = generate_spin_glass_sa(tanner, em, collect(T, 0:0.1:1.0), 1,false)
    ct = compile(IPDecoder(),tanner)
    res = decode(ct,syd)
    Eopt = sa_energy(vcat(res.error_qubits.xerror, res.error_qubits.zerror),prob)
	return Eopt,vcat(res.error_qubits.xerror, res.error_qubits.zerror)
end