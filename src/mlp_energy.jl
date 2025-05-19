"""
    MLPEnergyModel

A Multi-Layer Perceptron (MLP) model that maps a batch of input vectors to a single energy vector.

# Fields
- `input_dim::Int`: Dimension of the input vectors (N)
- `output_dim::Int`: Dimension of the output vector (M), calculated as N + N(N-1)/2
- `hidden_dims::Vector{Int}`: Dimensions of the hidden layers
- `model::Chain`: The Flux model chain
- `seed::Int`: Random seed for reproducibility

# Description
This model takes a batch of K vectors, each of length N, with elements in {-1, +1},
and outputs a single real-valued vector of length M, with values restricted to the range [-1, 1].
This output vector is used to define an energy function E, which maps the entire batch of input vectors to a scalar energy value.
The output dimension is calculated based on the input dimension.
"""
struct MLPEnergyModel
    input_dim::Int
    output_dim::Int
    hidden_dims::Vector{Int}
    model::Chain
    seed::Int

    function MLPEnergyModel(input_dim::Int, hidden_dims::Vector{Int}=Int[64, 32]; seed::Int=1234)
        Random.seed!(seed)

        # Calculate output dimension based on input_dim
        output_dim = Int(input_dim + input_dim * (input_dim - 1) / 2)

        # Build the model layers
        layers = []

        # Input layer to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims
            push!(layers, Dense(prev_dim => hidden_dim, relu))
            prev_dim = hidden_dim
        end

        # Final layer with tanh activation to restrict output to [-1, 1]
        push!(layers, Dense(prev_dim => output_dim, tanh))

        model = Chain(layers...)

        new(input_dim, output_dim, hidden_dims, model, seed)
    end

    # Constructor that accepts a model directly (needed for Zygote)
    function MLPEnergyModel(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int}, model::Chain, seed::Int)
        new(input_dim, output_dim, hidden_dims, model, seed)
    end
end

"""
    forward(model::MLPEnergyModel, x::AbstractMatrix)

Forward pass through the MLP model, aggregating across all input vectors.

# Arguments
- `model::MLPEnergyModel`: The MLP energy model
- `x::AbstractMatrix`: Input matrix of shape (N, K) where N is the input dimension and K is the batch size

# Returns
- `AbstractVector`: A single output vector of shape (M,) representing the entire batch
"""
function forward(model::MLPEnergyModel, x::AbstractMatrix)
    # Process each input vector separately
    batch_outputs = model.model(x)  # Shape: (M, K)

    # Aggregate across the batch dimension to get a single output vector
    # Using mean aggregation, but could use other methods like max, sum, etc.
    return vec(mean(batch_outputs, dims=2))  # Shape: (M,)
end

function calc_sample_energy(input_data::AbstractMatrix, input_dim, col_idx, linear_terms, quad_terms)
    # Ensure col_idx is within bounds
    @assert 1 <= col_idx <= size(input_data, 2) "Column index out of bounds: $col_idx not in 1:$(size(input_data, 2))"
    x_input = input_data[:, col_idx]

    # Linear term: ∑ hᵢ xᵢ
    E = vec(linear_terms)' * x_input

    # Quadratic term: ∑ Jᵢⱼ xᵢ xⱼ over i < j
    quad_sum = 0.0
    idx = 1
    for i in 1:input_dim-1
        for j in i+1:input_dim
            quad_sum += quad_terms[idx] * x_input[i] * x_input[j]
            idx += 1
        end
    end

    return E + quad_sum
end


function full_energy_function(model::MLPEnergyModel, pos_inputs::AbstractMatrix, neg_inputs::AbstractMatrix)
    # Get individual energies for positive and negative samples
    pos_energy, neg_energy = individual_energy_function(model, pos_inputs, neg_inputs)

    # Calculate mean energies
    mean_pos_energy = mean(pos_energy)
    mean_neg_energy = mean(neg_energy)

    # Return the difference (positive energy should be lower, negative energy should be higher)
    return mean_pos_energy - mean_neg_energy
end

function individual_energy_function(model::MLPEnergyModel, pos_inputs::AbstractMatrix, neg_inputs::AbstractMatrix)
    input_dim = model.input_dim

    # Combine positive and negative inputs
    all_inputs = hcat(pos_inputs, neg_inputs)
    num_pos = size(pos_inputs, 2)
    num_all = size(all_inputs, 2)

    # Process all inputs together
    all_output = forward(model, all_inputs)

    # Extract linear and quadratic terms for all samples
    linear_terms = all_output[1:input_dim, :]
    quad_terms = all_output[input_dim+1:end, :]

    # Calculate energy for each sample
    all_energies = [calc_sample_energy(all_inputs, input_dim, i, linear_terms, quad_terms) for i in 1:num_all]

    # Split into positive and negative energies
    pos_energy = all_energies[1:num_pos]
    neg_energy = all_energies[num_pos+1:end]

    return pos_energy, neg_energy
end


"""
    predict(model::MLPEnergyModel, x::AbstractMatrix)

Generate a single prediction vector for the entire batch of input vectors.

# Arguments
- `model::MLPEnergyModel`: The MLP energy model
- `x::AbstractMatrix`: Input matrix of shape (N, K) where N is the input dimension and K is the batch size

# Returns
- `AbstractVector`: A single output vector of shape (M,) representing the entire batch
"""
function predict(model::MLPEnergyModel, x::AbstractMatrix)
    return forward(model, x)
end

"""
    get_energy(model::MLPEnergyModel, pos_inputs::AbstractMatrix, neg_inputs::AbstractMatrix)

Calculate the energy value for the entire batch of input vectors.

# Arguments
- `model::MLPEnergyModel`: The MLP energy model
- `pos_inputs::AbstractMatrix`: Positive input matrix of shape (N, K₁) where N is the input dimension and K₁ is the number of positive samples
- `neg_inputs::AbstractMatrix`: Negative input matrix of shape (N, K₂) where N is the input dimension and K₂ is the number of negative samples

# Returns
- `Float64`: A single energy value for the entire batch, which is the difference between the mean positive and negative energies
"""
function get_energy(model::MLPEnergyModel, pos_inputs::AbstractMatrix, neg_inputs::AbstractMatrix)
    return full_energy_function(model, pos_inputs, neg_inputs)
end

function get_individual_energy(model::MLPEnergyModel, pos_inputs::AbstractMatrix, neg_inputs::AbstractMatrix)
    return individual_energy_function(model, pos_inputs, neg_inputs)
end

"""
    train!(model::MLPEnergyModel,
                        pos_inputs::AbstractMatrix,
                        neg_inputs::AbstractMatrix;
                        learning_rate=0.01,
                        epochs=100,
                        batch_size=16,
                        neg_weight=1.0)

Train the MLP model to minimize the energy of positive samples and maximize the energy of negative samples.
Uses a unified approach where positive and negative samples are combined into a single batch for forward pass.

# Arguments
- `model::MLPEnergyModel`: The MLP energy model
- `pos_inputs::AbstractMatrix`: Positive input matrix of shape (N, K₁) where N is the input dimension and K₁ is the number of positive samples
- `neg_inputs::AbstractMatrix`: Negative input matrix of shape (N, K₂) where N is the input dimension and K₂ is the number of negative samples
- `learning_rate::Float64`: Learning rate for the optimizer
- `epochs::Int`: Number of training epochs
- `batch_size::Int`: Batch size for training (total samples per batch)
- `neg_weight::Float64`: Weight for negative samples in the loss function

# Returns
- `Vector{Float64}`: Training loss history
"""
function train!(model::MLPEnergyModel,
                pos_inputs::AbstractMatrix,
                neg_inputs::AbstractMatrix;
                learning_rate=0.01,
                epochs=100,
                batch_size=16,
                neg_weight=1.0)
    # Ensure inputs are in the correct format (N, K)
    @assert size(pos_inputs, 1) == model.input_dim "Positive input dimension mismatch"
    @assert size(neg_inputs, 1) == model.input_dim "Negative input dimension mismatch"

    opt = Flux.setup(Flux.Adam(learning_rate), model.model)

    loss_history = Float64[]

    num_pos_samples = size(pos_inputs, 2)
    num_neg_samples = size(neg_inputs, 2)
    num_samples = num_pos_samples + num_neg_samples
    all_inputs = hcat(pos_inputs, neg_inputs)

    # Create labels: 1 for positive samples, 0 for negative samples
    labels = vcat(ones(Bool, num_pos_samples), zeros(Bool, num_neg_samples))

    @info "Training with $(num_pos_samples) positive samples and $(num_neg_samples) negative samples (batch size: $(batch_size))"

    for epoch in 1:epochs
        # Shuffle indices for each epoch
        indices = shuffle(1:num_samples)
        shuffled_inputs = all_inputs[:, indices]
        shuffled_labels = labels[indices]

        epoch_loss = 0.0
        num_batches = 0

        # Process in batches
        for i in 1:batch_size:num_samples
            batch_end = min(i + batch_size - 1, num_samples)
            batch_inputs = shuffled_inputs[:, i:batch_end]
            batch_labels = shuffled_labels[i:batch_end]

            if isempty(batch_labels)
                continue
            end

            # Get positive and negative indices within this batch
            pos_indices = findall(batch_labels)
            neg_indices = findall(.!batch_labels)

            if isempty(pos_indices) || isempty(neg_indices)
                continue
            end

            # Define a batch-specific loss function
            function batch_loss_fn(m)
                # Create a temporary model with the updated parameters
                temp_model = MLPEnergyModel(model.input_dim, model.output_dim, model.hidden_dims, m, model.seed)

                # Get model output for the entire batch
                batch_output = temp_model.model(batch_inputs)  # Shape: (output_dim, batch_size)

                # Calculate sample-specific energies using a functional approach
                input_dim = model.input_dim
                linear_terms = batch_output[1:input_dim, :]
                quad_terms = batch_output[input_dim+1:end, :]

                # Calculate energy for each sample using a functional approach
                # We'll create a function that calculates energy for a single sample
                function calc_sample_energy(col_idx)
                    x_input = batch_inputs[:, col_idx]

                    # Linear term: ∑ hᵢ xᵢ
                    E = sum(linear_terms[:, col_idx] .* x_input)

                    # Quadratic term: ∑ Jᵢⱼ xᵢ xⱼ over i < j
                    quad_sum = 0.0
                    idx = 1
                    for i in 1:input_dim-1
                        for j in i+1:input_dim
                            quad_sum += quad_terms[idx, col_idx] * x_input[i] * x_input[j]
                            idx += 1
                        end
                    end

                    return E + quad_sum
                end

                # Calculate energies for all samples
                sample_energies = [calc_sample_energy(col_idx) for col_idx in 1:size(batch_inputs, 2)]

                # Calculate average energy for positive and negative samples
                pos_energy = mean(sample_energies[pos_indices])
                neg_energy = mean(sample_energies[neg_indices])

                # Loss: minimize energy for positive samples, maximize for negative samples
                return pos_energy - neg_weight * neg_energy
            end

            loss, grads = Zygote.withgradient(batch_loss_fn, model.model)
            epoch_loss += loss
            num_batches += 1

            Flux.update!(opt, model.model, grads[1])

            push!(loss_history, loss)
        end

        if epoch % 10 == 0 || epoch == 1 && num_batches > 0
            @info "Epoch $epoch: Average Loss = $(epoch_loss / num_batches)"
        end
    end

    return loss_history
end
