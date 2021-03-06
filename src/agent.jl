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
    dt::Float64
    prev_v_::Float64
    prev_ω_::Float64
    estimator_::AbstractEstimator
    function EstimatorAgent(
        v::Float64,
        ω::Float64,
        dt::Float64,
        estimator::AbstractEstimator,
    )
        new(v, ω, dt, 0.0, 0.0, estimator)
    end
end

function decision(agent::Agent, observation::Nothing)
    return agent.v_, agent.ω_
end

function decision(agent::Agent, observation::Vector{Vector{Float64}})
    return agent.v_, agent.ω_
end

function decision(agent::EstimatorAgent, observation::Nothing)
    estimator = agent.estimator_
    motion_update(estimator, agent.prev_v_, agent.prev_ω_, agent.dt)
    agent.prev_v_, agent.prev_ω_ = agent.v_, agent.ω_
    return agent.v_, agent.ω_
end

function decision(
    agent::EstimatorAgent,
    observation::Vector{Vector{Float64}},
    envmap::Map;
    resample = true,
    sensor_reset = false,
)
    estimator = agent.estimator_
    motion_update(estimator, agent.prev_v_, agent.prev_ω_, agent.dt)
    agent.prev_v_, agent.prev_ω_ = agent.v_, agent.ω_
    if typeof(agent.estimator_) == Mcl || typeof(agent.estimator_) == KldMcl
        observation_update(agent.estimator_, observation, envmap; resample = resample)
    elseif typeof(agent.estimator_) == ResetMcl
        observation_update(
            agent.estimator_,
            observation,
            envmap;
            resample = resample,
            sensor_reset = sensor_reset,
        )
    elseif typeof(agent.estimator_) == AMcl
        observation_update(agent.estimator_, observation, envmap)
    elseif typeof(agent.estimator_) == KalmanFilter
        observation_update(agent.estimator_, observation)
    end
    return agent.v_, agent.ω_
end

function draw(agent::Agent, p::Plot{T}) where {T} end

function draw(agent::EstimatorAgent, p::Plot{T}) where {T}
    draw(agent.estimator_, p)
end
