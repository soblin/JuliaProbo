mutable struct DpPolicyAgent <: AbstractAgent
    puddle_ignore_agent_::PuddleIgnoreAgent
    reso::Vector{Float64}
    pose_min::Vector{Float64}
    pose_max::Vector{Float64}
    index_nums::Vector{Int64}
    policy_data::AbstractArray{Float64,4}
end

function DpPolicyAgent(
    agent::PuddleIgnoreAgent,
    reso::Vector{Float64};
    lowerleft = [-4.0, -4.0],
    upperright = [4.0, 4.0],
    fname = "policy.txt",
)
    pose_min = vcat(lowerleft, [0.0])
    pose_max = vcat(upperright, [2pi])
    index_nums = [convert(Int64, round((pose_max[i] - pose_min[i]) / reso[i])) for i = 1:3]
    policy_data = zeros(Float64, index_nums..., 2)
    fd = open(fname, "r")
    lines = readlines(fd)
    for line in lines
        tokens = split(line, " ")
        tokens = map(x -> parse(Float64, x), tokens)
        id1, id2, id3 = map(x -> convert(Int64, x), tokens[1:3])
        v, ω = tokens[4], tokens[5]
        policy_data[id1, id2, id3, :] = [v, ω]
    end
    return DpPolicyAgent(agent, reso, pose_min, pose_max, index_nums, policy_data)
end

function policy(agent::DpPolicyAgent, pose::Vector{Float64}, goal = nothing)
    if agent.puddle_ignore_agent_.in_goal_
        return [0.0, 0.0]
    end
    reso = agent.reso
    pose_min = agent.pose_min
    index_nums = agent.index_nums
    index = [convert(Int64, round((pose[i] - pose_min[i]) / reso[i])) for i = 1:3]
    index[3] = (index[3] + 10 * index_nums[3]) % index_nums[3]
    if index[3] == 0
        index[3] = index_nums[3]
    end
    for i in [1, 2]
        if index[i] < 0
            index[i] = 0
        end
        if index[i] > index_nums[i]
            index[i] = index_nums[i]
        end
    end
    return agent.policy_data[index..., :]
end

function decision(
    dp_agent::DpPolicyAgent,
    observation::Vector{Vector{Float64}},
    envmap = Map();
    kwargs...,
)
    agent = dp_agent.puddle_ignore_agent_
    if agent.in_goal_
        return 0.0, 0.0
    end
    motion_update(agent.estimator_, agent.prev_v_, agent.prev_ω_, agent.dt)
    observation_update(agent.estimator_, observation, envmap; kwargs...)
    agent.total_reward_ += agent.dt * reward_per_sec(agent)

    v, ω = policy(dp_agent, agent.estimator_.pose_, agent.goal)
    agent.prev_v_, agent.prev_ω_ = v, ω
    return v, ω
end
