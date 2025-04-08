abstract type OptimizerType end
struct AdamOpt <: OptimizerType 
    learning_rate::Real
    β1::Real
    β2::Real

    function AdamOpt(learning_rate::Real=0.001, β1::Real=0.9, β2::Real=0.999)
        return new(learning_rate, β1, β2)
    end
end
struct RMSpropOpt <: OptimizerType 
    learning_rate::Real
    ρ::Real
    
    function RMSpropOpt(learning_rate::Real=0.001, ρ::Real=0.9)
        return new(learning_rate, ρ)
    end
end

function get_optimizer(opt_type::OptimizerType)
    if opt_type isa AdamOpt
        return Flux.Adam(opt_type.learning_rate, (opt_type.β1, opt_type.β2))
    elseif opt_type isa RMSpropOpt
        return Flux.RMSProp(opt_type.learning_rate, opt_type.ρ)
    else
        error("Unknown optimizer type.")
    end
end

# -----------------------------
abstract type AnnealingStrategy end
struct LinearAnnealing <: AnnealingStrategy end
struct ExponentialAnnealing <: AnnealingStrategy end
struct InverseAnnealing <: AnnealingStrategy end

function get_betas(strategy::AnnealingStrategy, num_steps, betamin, betamax)
    if strategy isa LinearAnnealing
        return range(betamin, betamax; length=num_steps)
    elseif strategy isa ExponentialAnnealing
        return exp.(range(log(betamin), log(betamax); length=num_steps))
    elseif strategy isa InverseAnnealing
        return 1 ./ range(betamax, betamin; length=num_steps)
    else
        error("Unknown annealing strategy.")
    end
end
