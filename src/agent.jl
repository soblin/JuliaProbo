using JuliaProbo

mutable struct Agent <: AbstractAgent
    v_::Float64
    ω_::Float64
    function Agent(v::Float64, ω::Float64)
        new(v, ω)
    end
end

mutable struct EstimatorAgent <: AbstractAgent
    v_::Float64
    ω_::Float64
    estimator_::AbstractEstimator
    function EstimatorAgent(v::Float64, ω::Float64, estimator::AbstractEstimator)
        new(v, ω, estimator)
    end
end

function decision(agent::Agent, observation::Nothing)
    return agent.v_, agent.ω_
end

function decision(agent::Agent, observation::Vector{Vector{Float64}})
    return agent.v_, agent.ω_
end

function decision(agent::EstimatorAgent, observation::Vector{Vector{Float64}})
    return agent.v_, agent.ω_
end

function draw(agent::Agent, p::Plot{T}) where T end

function draw(agent::EstimatorAgent, p::Plot{T}) where T
    draw(agent.estimator_, p)
end
