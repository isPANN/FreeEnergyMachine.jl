"""
A mutable struct representing a simulated bifurcation system.
T: Type parameter for numerical values
KIND: Type of bifurcation (:aSB, :bSB, or :dSB)
"""
mutable struct SimulatedBifurcation{T, KIND, P<:CombinatorialProblem}
    a::T                    # Control parameter that varies during simulation
    const c0::T             # Coupling strength
    const problem::P
    const clamp::Bool

    function SimulatedBifurcation{KIND}(a, c0, problem::P) where {KIND, P}
        @assert KIND in (:aSB, :bSB, :dSB) "Invalid bifurcation type: $KIND, must be one of (:aSB, :bSB, :dSB)"
        T = problem.dtype
        if KIND == :aSB
            clamp = false
        else
            clamp = true
        end
        return new{T, KIND, P}(T(a), T(c0), problem, clamp)
    end
end

function SimulatedBifurcation{KIND}(problem::P; c0=-1.0) where {KIND, P<:CombinatorialProblem}
    @assert KIND in (:aSB, :bSB, :dSB) "Invalid bifurcation type: $KIND, must be one of (:aSB, :bSB, :dSB)"
    T = problem.dtype
    if c0 < 0
        c0 = 0.5/sqrt(problem.node_num)/norm(problem.coupling)
    end
    return SimulatedBifurcation{KIND}(T(1.0), T(c0), problem)
end

# -----------------------
# Calculate potential energy
# -----------------------
"""
Calculate potential energy for asymmetric simulated bifurcation (aSB).
V = Σ(x^4/4 + ax^2/2) - c0 Σ(Jij xi xj)
"""
function potential_energy(sys::SimulatedBifurcation{T, :aSB}, x::Matrix{T}) where T
    node_num, batch_size = size(x)
    V = zeros(T, batch_size)

    for b in 1:batch_size
        V1 = zero(T)
        for i in 1:node_num
            xi = x[i, b]
            V1 += xi^4 / 4 + (sys.a / 2) * xi^2
        end

        V2 = zero(T)
        for i in 1:node_num
            for j in i+1:node_num
                J = sys.problem.coupling[i, j]
                J == 0 && continue
                V2 += J * x[i, b] * x[j, b]
            end
        end
        V[b] = V1 - sys.c0 * V2
    end

    return V
end


"""
Calculate potential energy for symmetric simulated bifurcation (bSB).
V = Σ(ax^2/2) - c0 Σ(Jij xi xj)
"""
function potential_energy(sys::SimulatedBifurcation{T, :bSB}, x::Matrix{T}) where T
    node_num, batch_size = size(x)
    V = zeros(T, batch_size)

    for b in 1:batch_size
        V1 = zero(T)
        for i in 1:node_num
            V1 += (sys.a / 2) * x[i, b]^2
        end

        V2 = zero(T)
        for i in 1:node_num
            for j in i+1:node_num
                J = sys.problem.coupling[i, j]
                J == 0 && continue
                V2 += J * x[i, b] * x[j, b]
            end
        end
        V[b] = V1 - sys.c0 * V2
    end

    return V
end

"""
Calculate potential energy for discrete simulated bifurcation (dSB).
V = Σ(ax^2/2) - c0 Σ(Jij (xi sign(xj) + xj sign(xi)))
"""
function potential_energy(sys::SimulatedBifurcation{T, :dSB}, x::Matrix{T}) where T
    node_num, batch_size = size(x)
    V = zeros(T, batch_size)

    for b in 1:batch_size
        V1 = zero(T)
        for i in 1:node_num
            V1 += (sys.a / 2) * x[i, b]^2
        end

        V2 = zero(T)
        for i in 1:node_num
            for j in i+1:node_num
                J = sys.problem.coupling[i, j]
                J == 0 && continue
                V2 += J * (x[i, b] * sign(x[j, b]) + x[j, b] * sign(x[i, b]))
            end
        end

        V[b] = V1 - sys.c0 * V2
    end
    return V
end

# -----------------------
# Calculate kinetic energy
# -----------------------

"""Calculate kinetic energy: K = Σ(p^2/2)"""
kinetic_energy(::SimulatedBifurcation{T}, p::Matrix{T}) where T = sum(p .^ 2, dims=1)[:] ./ 2

# -----------------------
# Calculate forces
# (This term is used to update the momentum.)
# -----------------------
"""
Calculate forces for asymmetric simulated bifurcation (aSB).
F = -(x^3 + ax) + c0 Σ(Jij xj)
"""
function force!(f::Matrix{T}, sys::SimulatedBifurcation{T, :aSB}, x::Matrix{T}) where T
    @. f = -x^3 - sys.a * x
    f .+= sys.c0 .* (sys.problem.coupling * x)
    return f
end
function force!(f::Matrix{T}, sys::SimulatedBifurcation{T, :bSB}, x::Matrix{T}) where T
    @. f = -sys.a * x
    f .+= sys.c0 .* (sys.problem.coupling * x)
    return f
end
function force!(f::Matrix{T}, sys::SimulatedBifurcation{T, :dSB}, x::Matrix{T}) where T
    @. f = -sys.a * x
    f .+= sys.c0 .* (sys.problem.coupling * sign.(x))
    return f
end

# Helper functions to create force arrays of appropriate size
force(sys::SimulatedBifurcation{T}, x::Matrix{T}) where T = force!(Matrix{T}(undef, size(x)), sys, x)


"""
State of the simulated bifurcation system.
x: Position variables
p: Momentum variables
"""
struct SimulatedBifurcationState{T}
    x::Matrix{T}
    p::Matrix{T}
    function SimulatedBifurcationState(x::Matrix{T}, p::Matrix{T}) where T
        @assert size(x) == size(p) "Position and momentum matrices must have the same dimensions"
        return new{T}(x, p)
    end
end

# Constructor for single instance
# function SimulatedBifurcationState(length::Int, initial_scale=0.05; dtype=Float32)
#     return SimulatedBifurcationState(randn(dtype, length).*dtype(initial_scale), randn(dtype, length).*dtype(initial_scale))
# end

# Constructor for batch processing
function SimulatedBifurcationState(batch_size::Int, length::Int, initial_scale=0.05; dtype=Float32)
    return SimulatedBifurcationState(randn(dtype,  length, batch_size).*dtype(initial_scale), randn(dtype, length, batch_size).*dtype(initial_scale))
end

"""
Checkpoint structure to store system state during simulation.
"""
struct SBCheckpoint{T}
    a::T                                # Control parameter value
    time::T                             # Simulation time
    potential_energy::Union{T, Vector{T}}  # Potential energy at checkpoint (scalar for single instance, vector for batch)
    kinetic_energy::Union{T, Vector{T}}    # Kinetic energy at checkpoint (scalar for single instance, vector for batch)
    state::SimulatedBifurcationState{T}    # Copy of system state
end

"""
Simulate the bifurcation system using Störmer-Verlet integrator.

Parameters:
- state: Initial state of the system
- sys: Bifurcation system
- nsteps: Number of simulation steps
- dt: Time step size
- a0: Initial control parameter value (default: 1.0)
- a1: Final control parameter value (default: 0.0)
- checkpoint_steps: Number of steps between checkpoints (default: typemax(Int))

Returns:
- Updated state
- Vector of checkpoints
"""
function simulate_bifurcation!(state::SimulatedBifurcationState{T},
                             sys::SimulatedBifurcation{T};
                             nsteps::Int, dt, a0=1.0, a1=0.0,
                             checkpoint_steps::Int=typemax(Int)) where T
    checkpoints = Vector{SBCheckpoint{T}}()

    # Initial half-step for momentum (Störmer-Verlet initialization)
    sys.a = a0
    f = force(sys, state.x)
    state.p .+= 0.5 * dt * f

    for i in 1:nsteps
        # Position update
        state.x .+= dt * state.p

        # Apply position constraints if clamping is enabled
        if sys.clamp
            batch_size = size(state.x, 2)
            for b in 1:batch_size, j in 1:sys.problem.node_num
                if state.x[j, b] > 1
                    state.x[j, b] = 1
                    state.p[j, b] = 0  # Reset momentum at boundaries
                elseif state.x[j, b] < -1
                    state.x[j, b] = -1
                    state.p[j, b] = 0
                end
            end
        end

        # Update control parameter and calculate forces
        sys.a = a0 + (a1 - a0) * i / nsteps
        force!(f, sys, state.x)

        # Momentum update (Störmer-Verlet step)
        if i < nsteps
            state.p .+= dt * f
        else
            # Final half-step for momentum
            state.p .+= 0.5 * dt * f
        end

        # Save checkpoint if needed
        if mod(i, checkpoint_steps) == 0
            push!(checkpoints, SBCheckpoint{T}(
                sys.a,
                i * dt,
                potential_energy(sys, state.x),
                kinetic_energy(sys, state.p),
                SimulatedBifurcationState{T}(copy(state.x), copy(state.p))
            ))
        end
    end

    return state, checkpoints
end