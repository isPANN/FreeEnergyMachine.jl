abstract type AnnealingStrategy end
struct LinearAnnealing <: AnnealingStrategy end
struct ExponentialAnnealing <: AnnealingStrategy end
struct InverseAnnealing <: AnnealingStrategy end

function get_betas(::LinearAnnealing, num_steps, betamin, betamax)
    return range(betamin, betamax; length=num_steps)
end

function get_betas(::ExponentialAnnealing, num_steps, betamin, betamax)
    return exp.(range(log(betamin), log(betamax); length=num_steps))
end

function get_betas(::InverseAnnealing, num_steps, betamin, betamax)
    return 1 ./ range(betamax, betamin; length=num_steps)
end