# Full adder function (handles single bit and carry)
function full_adder(A::Int, B::Int, carry_in::Int)
    sum_bit = A ⊻ B ⊻ carry_in  # XOR operation -> Addition
    carry_in = (A & B) | (B & carry_in) | (A & carry_in)  # AND operation -> Carry
    return sum_bit, carry_in  # Return sum and carry
end

# N-bit adder
function n_bit_adder(A::Vector{Int}, B::Vector{Int})
    N = length(A)  # Get bit length N
    sum_result = Int[]  # Store sum result
    carry_in = 0  # Carry input for the lowest bit is 0

    # Start from the lowest bit (right side)
    for i in N:-1:1
        sum_bit, carry_in = full_adder(A[i], B[i], carry_in)  # Execute full adder for each bit
        pushfirst!(sum_result, sum_bit)  # Insert sum to the front of the result
    end

    pushfirst!(sum_result, carry_in)

    result = vcat(A, B, sum_result)

    return result  # Return full vector containing A, B, and sum
end

function generate_dataset(N::Int; check_errors=true, ancilla_bits=0)
    patterns = Vector{Vector{Int}}()

    # Generate all possible N-bit binary numbers
    all_binary_numbers = Vector{Int}[]
    for i in 0:(2^N - 1)
        binary = digits(i, base=2, pad=N)
        reverse!(binary)
        push!(all_binary_numbers, binary)
    end

    for A in all_binary_numbers
        for B in all_binary_numbers
            result = n_bit_adder(A, B)
            push!(patterns, result)
        end
    end

    if !isempty(patterns)
        len = length(patterns[1])
        for pattern in patterns
            if length(pattern) != len
                error("All patterns must have the same length.")
            end
        end

        # Expand with ancilla bits if needed
        if ancilla_bits > 0
            ancilla_combinations = collect(Iterators.product(ntuple(_ -> (0, 1), ancilla_bits)...))
            new_patterns = Vector{Vector{Int}}()
            for p in patterns
                for ancilla in ancilla_combinations
                    push!(new_patterns, vcat(p, collect(ancilla)))
                end
            end
            patterns = new_patterns
        end

        if check_errors
            if len + ancilla_bits > 20
                @warn "Checking error patterns for len=$(len + ancilla_bits) may consume a lot of memory."
            end

            all_patterns = [digits(i, base=2, pad=len+ancilla_bits) |> reverse for i in 0:(2^(len+ancilla_bits) - 1)]
            error_patterns = setdiff(all_patterns, patterns)

            return hcat(patterns...), hcat(error_patterns...)
        end
    end

    return hcat(patterns...)
end

function energy_value(J, K, h, x::Vector{Int})
    energy = 0
    N = length(x)
    for i in 1:N
        for j in 1:N
            for k in 1:N
                energy += K[i, j, k] * x[i] * x[j] * x[k]
            end
            energy += J[i, j] * x[i] * x[j]
        end
        energy += h[i] * x[i]
    end
    # for i in 1:N
    #     for j in 1:N
    #         energy += J[i, j] * x[i] * x[j]
    #     end
    #     energy += h[i] * x[i]
    # end
    return energy
end


function my_optimise(N, patterns, error_patterns, optimizer, env)
    patterns = map(pattern -> 1 .- 2 .* pattern, patterns)
    error_patterns = map(pattern -> 1 .- 2 .* pattern, error_patterns)

    opt = isnothing(env) ? optimizer() : optimizer(env)
    model = direct_model(opt)
    set_silent(model)
    set_string_names_on_creation(model, false)

    @variable(model, -1 <= J[1:N, 1:N] <= 1)
    @variable(model, -1 <= K[1:N, 1:N, 1:N] <= 1)
    @variable(model, -1 <= h[1:N] <= 1)

    @objective(model, Min,
        sum(energy_value(J, K, h, pattern) for pattern in patterns) / length(patterns) - sum(energy_value(J, K, h, pattern) for pattern in error_patterns) / length(error_patterns))

    optimize!(model)

    J_opt = value.(J)
    K_opt = value.(K)
    h_opt = value.(h)
    @show J_opt, K_opt, h_opt
    @info [energy_value(J_opt, K_opt, h_opt, pattern) for pattern in patterns]
    @info [energy_value(J_opt, K_opt, h_opt, pattern) for pattern in error_patterns]
    return J_opt, K_opt, h_opt, [energy_value(J_opt, K_opt, h_opt, pattern) for pattern in patterns], [energy_value(J_opt, K_opt, h_opt, pattern) for pattern in error_patterns]
end

