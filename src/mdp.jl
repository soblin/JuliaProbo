mutable struct PuddleIgnoreAgent <: AbstractAgent
    v_::Float64
    ω_::Float64
    dt::Float64
    prev_v_::Float64
    prev_ω_::Float64
    estimator_::AbstractEstimator
    puddle_coeff::Float64
    puddle_depth_::Float64
    total_reward_::Float64
    in_goal_::Bool
    final_value_::Float64
    goal::Goal
    function PuddleIgnoreAgent(
        v::Float64,
        ω::Float64,
        dt::Float64,
        estimator::AbstractEstimator,
        goal::Goal;
        puddle_coeff = 100,
    )
        new(v, ω, dt, 0.0, 0.0, estimator, puddle_coeff, 0.0, 0.0, false, 0.0, goal)
    end
end

function reward_per_sec(agent::PuddleIgnoreAgent)
    return -1.0 - agent.puddle_depth_ * agent.puddle_coeff
end

function policy(agent::PuddleIgnoreAgent, goal::Goal)
    cur_bel_pose = agent.estimator_.pose_
    x, y, θ = cur_bel_pose[1], cur_bel_pose[2], cur_bel_pose[3]
    dx, dy = goal.x - x, goal.y - y
    direction = convert(Int64, round((atan(dy, dx) - θ) * 180 / pi))
    while direction > 180
        direction -= 360
    end
    while direction <= -180
        direction += 360
    end
    v, ω = 0.0, 0.0
    if direction > 10
        v, ω = 0.0, 2.0
    elseif direction < -10
        v, ω = 0.0, -2.0
    else
        v, ω = 1.0, 0.0
    end
    v, ω
end

function decision(agent::PuddleIgnoreAgent, observation::Vector{Vector{Float64}})
    if agent.in_goal_
        return 0.0, 0.0
    end
    motion_update(agent.estimator_, agent.prev_v_, agent.prev_ω_, agent.dt)
    obseravtion = Vector{Vector{Float64}}(undef, 0)
    observation_update(agent.estimator_, observation)
    agent.total_reward_ += agent.dt * reward_per_sec(agent)

    v, ω = policy(agent, agent.goal)
    agent.prev_v_, agent.prev_ω_ = v, ω
    return v, ω
end

function draw(agent::PuddleIgnoreAgent, p::Plot{T}) where {T}
    x, y = agent.estimator_.pose_[1], agent.estimator_.pose_[2]
    annota1 = "reward/sec: $(round(reward_per_sec(agent), sigdigits=3))"
    annota2 = "total reward: $(round(agent.total_reward_ + agent.final_value_, sigdigits=3))"
    p = annotate!(x + 1.0, y - 0.5, text(annota1, 10))
    p = annotate!(x + 1.0, y - 1.0, text(annota2, 10))
end
