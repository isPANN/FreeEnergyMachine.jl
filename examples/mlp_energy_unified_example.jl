using FreeEnergyMachine
using Random
using Plots
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Get dataset
patterns, error_patterns = generate_dataset(2; check_errors = true, ancilla_bits=0)

# Parameters
input_dim = size(patterns, 1)   # N: length of input vectors
num_pos_samples = size(patterns, 2) # Number of positive samples
num_neg_samples = size(error_patterns, 2) # Number of negative samples
hidden_dims = [128, 64]  # Hidden layer dimensions

println("Dataset information:")
println("Positive samples: $num_pos_samples")
println("Negative samples: $num_neg_samples")

# Generate positive and negative input data with values in {-1, +1}
pos_inputs = 2.0f0 .* patterns .- 1.0f0
neg_inputs = 2.0f0 .* error_patterns .- 1.0f0

# Create the MLP energy model
model = MLPEnergyModel(input_dim, hidden_dims; seed=1234)

# Calculate initial energies
initial_energy = get_energy(model, pos_inputs, neg_inputs)
println("Initial energy: ", initial_energy)


loss_history = train!(model, pos_inputs, neg_inputs;
                    learning_rate=0.01,
                    epochs=200,
                    batch_size=50,
                    neg_weight=1)

# Calculate final energies
final_energy = get_energy(model, pos_inputs, neg_inputs)
println("Final energy: ", final_energy)
println("Energy reduction: ", initial_energy - final_energy)

# Plot the loss history
p = plot(loss_history,
     title="Training Loss",
     xlabel="Iteration",
     ylabel="Energy Difference",
     legend=false,
     linewidth=2)

# Save the plot
savefig(p, "mlp_energy_unified_training.png")

# Combine positive and negative samples for testing
all_inputs = hcat(pos_inputs, neg_inputs)
true_labels = vcat(ones(Bool, num_pos_samples), zeros(Bool, num_neg_samples))

# Get individual energy values for each sample using the new function
pos_energies, neg_energies = get_individual_energy(model, pos_inputs, neg_inputs)

# Combine positive and negative energies for further analysis
sample_energies = vcat(pos_energies, neg_energies)

# Find the best threshold
thresholds = range(minimum(sample_energies), maximum(sample_energies), length=100)

# Use a function to avoid scope issues
function find_best_threshold(thresholds, sample_energies, true_labels)
    best_accuracy = 0.0
    best_threshold = 0.0

    for t in thresholds
        pred = sample_energies .<= t
        acc = mean(pred .== true_labels)
        if acc > best_accuracy
            best_accuracy = acc
            best_threshold = t
        end
    end

    return best_threshold, best_accuracy
end

# Call the function to get the best threshold and accuracy
best_threshold, best_accuracy = find_best_threshold(thresholds, sample_energies, true_labels)

println("\nClassification results:")
println("Best threshold: $best_threshold")
println("Best accuracy: $(round(best_accuracy * 100, digits=2))%")

# Plot energy distribution
# We already have pos_energies and neg_energies from get_individual_energy

p1 = histogram(pos_energies,
          bins=10,
          alpha=0.5,
          label="Positive Samples",
          title="Energy Distribution",
          xlabel="Energy",
          ylabel="Count")
histogram!(p1, neg_energies, bins=10, alpha=0.5, label="Negative Samples")
vline!([best_threshold], label="Best Threshold", linewidth=2, color=:black, linestyle=:dash)

# Save the energy distribution plot
savefig(p1, "mlp_energy_unified_distribution.png")

# Print detailed results
println("\nDetailed results:")
println("Sample\tType\t\tEnergy")
println("------\t----\t\t------")

# Print positive samples
for i in 1:min(5, length(pos_energies))
    println("$i\tPositive\t$(round(pos_energies[i], digits=4))")
end

# Print negative samples
for i in 1:min(5, length(neg_energies))
    println("$(i+length(pos_energies))\tNegative\t$(round(neg_energies[i], digits=4))")
end

# Print average energies
println("\nAverage positive energy: $(round(mean(pos_energies), digits=4))")
println("Average negative energy: $(round(mean(neg_energies), digits=4))")
println("Energy difference: $(round(mean(neg_energies) - mean(pos_energies), digits=4))")
