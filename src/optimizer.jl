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

get_optimizer(opt::AdamOpt) = Flux.Adam(opt.learning_rate, (opt.β1, opt.β2))
get_optimizer(opt::RMSpropOpt) = Flux.RMSProp(opt.learning_rate, opt.ρ)

# -----------------------------
abstract type AnnealingStrategy end
struct LinearAnnealing <: AnnealingStrategy end
struct ExponentialAnnealing <: AnnealingStrategy end
struct InverseAnnealing <: AnnealingStrategy end

get_betas(::LinearAnnealing, num_steps, betamin, betamax) = range(betamin, betamax; length=num_steps)
get_betas(::ExponentialAnnealing, num_steps, betamin, betamax) = exp.(range(log(betamin), log(betamax); length=num_steps))
get_betas(::InverseAnnealing, num_steps, betamin, betamax) = 1 ./ range(betamax, betamin; length=num_steps)
