mutable struct Agent <: AbstractAgent
    v_::Float64
    ω_::Float64
    function Agent(v::Float64, ω::Float64)
        new(v, ω)
    end
end

function decision(agent::Agent, observation::Nothing)
    return agent.v_, agent.ω_
end

function decision(agent::Agent, observation::Vector{Vector{Float64}})
    return agent.v_, agent.ω_
end

function draw(agent::Agent, p::Plot{T}) where {T} end

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
    kwargs...,
)
    estimator = agent.estimator_

    motion_update(
        estimator,
        agent.prev_v_,
        agent.prev_ω_,
        agent.dt;
        kwargs...,
        observation = observation,
    )

    agent.prev_v_, agent.prev_ω_ = agent.v_, agent.ω_

    observation_update(agent.estimator_, observation, envmap; kwargs...)

    return agent.v_, agent.ω_
end

function draw(agent::EstimatorAgent, p::Plot{T}) where {T}
    draw(agent.estimator_, p)
end

mutable struct LoggerAgent <: AbstractAgent
    v_::Float64
    ω_::Float64
    dt::Float64
    pose_::Vector{Float64}
    step_::Int64
    log_::IOStream
    function LoggerAgent(
        v::Float64,
        ω::Float64,
        dt::Float64,
        init_pose::Vector{Float64},
        fname = "log.txt",
    )
        fd = open(fname, "w")
        write(fd, "delta $(dt)\n")
        new(v, ω, dt, copy(init_pose), 0, fd)
    end
end

function decision(agent::LoggerAgent, observation::Vector{Vector{Float64}})
    write(agent.log_, "u $(agent.step_) $(agent.v_) $(agent.ω_)\n")
    pose = agent.pose_
    write(agent.log_, "x $(agent.step_) $(pose[1]) $(pose[2]) $(pose[3])\n")
    N = size(observation)[1]
    for i = 1:N
        obsv = observation[i]
        # obsv = [idx, d, ϕ, ψ]
        write(
            agent.log_,
            "z $(agent.step_) $(convert(Int64, obsv[4])) $(obsv[1]) $(obsv[2]) $(obsv[3])\n",
        )
    end

    agent.step_ += 1
    flush(agent.log_)
    agent.pose_ = state_transition(agent.pose_, agent.v_, agent.ω_, agent.dt)
    return agent.v_, agent.ω_
end

function draw(agent::LoggerAgent, p::Plot{T}) where {T} end
