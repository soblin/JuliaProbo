mutable struct DpPolicyAgent <: AbstractMDPAgent
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
    reso::Vector{Float64}
    pose_min::Vector{Float64}
    pose_max::Vector{Float64}
    index_nums::Vector{Int64}
    policy_data::AbstractArray{Float64,4}
end

function DpPolicyAgent(
    v::Float64,
    ω::Float64,
    dt::Float64,
    estimator::AbstractEstimator,
    goal::Goal,
    reso::Vector{Float64};
    lowerleft = [-4.0, -4.0],
    upperright = [4.0, 4.0],
    puddle_coeff = 100,
)
    pose_min = vcat(lowerleft, [0.0])
    pose_max = vcat(upperright, [2pi])
    index_nums = [convert(Int64, round((pose_max[i] - pose_min[i]) / reso[i])) for i = 1:3]
    policy_data = zeros(Float64, index_nums..., 2)
    return DpPolicyAgent(
        v,
        ω,
        dt,
        0.0,
        0.0,
        estimator,
        puddle_coeff,
        0.0,
        0.0,
        false,
        0.0,
        goal,
        reso,
        pose_min,
        pose_max,
        index_nums,
        policy_data,
    )
end

function init_policy(agent::DpPolicyAgent, fname = "policy.txt")
    policy_data = agent.policy_data
    fd = open(fname, "r")
    lines = readlines(fd)
    for line in lines
        tokens = split(line, " ")
        tokens = map(x -> parse(Float64, x), tokens)
        id1, id2, id3 = map(x -> convert(Int64, x), tokens[1:3])
        v, ω = tokens[4], tokens[5]
        policy_data[id1, id2, id3, :] = [v, ω]
    end
end

function policy(agent::DpPolicyAgent)
    if agent.in_goal_
        return [0.0, 0.0]
    end
    reso = agent.reso
    pose = agent.estimator_.pose_
    pose_min = agent.pose_min
    index_nums = agent.index_nums
    index = [convert(Int64, round((pose[i] - pose_min[i]) / reso[i])) for i = 1:3]
    index[3] = (index[3] + 10 * index_nums[3]) % index_nums[3]
    if index[3] == 0
        index[3] = index_nums[3]
    end
    for i in [1, 2]
        if index[i] < 1
            index[i] = 1
        end
        if index[i] > index_nums[i]
            index[i] = index_nums[i]
        end
    end
    return agent.policy_data[index..., :]
end

mutable struct QMDPAgent <: AbstractMDPAgent
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
    pose_min::Vector{Float64}
    pose_max::Vector{Float64}
    dp::PolicyEvaluator
    current_value_::Float64
end

function QMDPAgent(
    v::Float64,
    ω::Float64,
    dt::Float64,
    estimator::AbstractEstimator,
    goal::Goal,
    reso::Vector{Float64};
    lowerleft = [-4.0, -4.0],
    upperright = [4.0, 4.0],
    puddle_coeff = 100,
)
    pose_min = vcat(lowerleft, [0.0])
    pose_max = vcat(upperright, [2pi])
    dp = PolicyEvaluator(
        reso,
        goal;
        lowerleft = lowerleft,
        upperright = upperright,
        dt = dt,
        puddle_coeff = puddle_coeff,
    )
    return QMDPAgent(
        v,
        ω,
        dt,
        0.0,
        0.0,
        estimator,
        puddle_coeff,
        0.0,
        0.0,
        false,
        0.0,
        goal,
        pose_min,
        pose_max,
        dp,
        0.0,
    )
end

function init_policy(agent::QMDPAgent, fname = "policy.txt")
    # need this to set `indices` in PolicyEvaluator
    init_value(agent.dp)
    policy_data = agent.dp.policy_
    fd = open(fname, "r")
    lines = readlines(fd)
    for line in lines
        tokens = split(line, " ")
        tokens = map(x -> parse(Float64, x), tokens)
        id1, id2, id3 = map(x -> convert(Int64, x), tokens[1:3])
        v, ω = tokens[4], tokens[5]
        policy_data[id1, id2, id3, :] = [v, ω]
    end
    init_state_transition_probs(agent.dp, agent.dp.dt, 10)
end

function init_value(agent::QMDPAgent, fname = "value.txt")
    value_data = agent.dp.value_function_
    fd = open(fname, "r")
    lines = readlines(fd)
    for line in lines
        tokens = split(line, " ")
        tokens = map(x -> parse(Float64, x), tokens)
        id1, id2, id3 = map(x -> convert(Int64, x), tokens[1:3])
        v = tokens[4]
        value_data[id1, id2, id3] = v
    end
end

function correct_index(index_::Vector{Int64}, index_nums::Vector{Int64})
    index = copy(index_)
    while index[3] > index_nums[3]
        index[3] -= index_nums[3]
    end
    while index[3] < 0
        index[3] += index_nums[3]
    end
    if index[3] == 0
        index[3] = index_nums[3]
    end
    for i in [1, 2]
        if index[i] < 1
            index[i] = 1
        end
        if index[i] > index_nums[i]
            index[i] = index_nums[i]
        end
    end
    return index
end

function to_index(
    pose::Vector{Float64},
    pose_min::Vector{Float64},
    reso::Vector{Float64},
    index_nums::Vector{Int64},
)
    index = [convert(Int64, round((pose[i] - pose_min[i]) / reso[i])) for i = 1:3]
    return correct_index(index, index_nums)
end

function action_value(agent::QMDPAgent, action::Vector{Float64}, index::Vector{Int64})
    v, ω = action[1], action[2]
    value = 0.0
    dp = agent.dp
    state_transition_probs = dp.state_transition_probs
    value_function = dp.value_function_
    index_nums = agent.dp.index_nums
    for trans_probs in state_transition_probs[(v, ω, index[3])]
        trans_ind = trans_probs[1]
        prob = trans_probs[2]
        after_ = [index...] .+ trans_ind
        after = correct_index(after_, index_nums)
        reward = -dp.dt * dp.depth[after[1], after[2]] * dp.puddle_coeff - dp.dt
        value += (value_function[after...] + reward) * prob
    end
    return value
end

function evaluation(
    agent::QMDPAgent,
    action::Vector{Float64},
    indices::Vector{Vector{Int64}},
)
    return sum([action_value(agent, action, index) for index in indices]) / length(indices)
end

function policy(agent::QMDPAgent)
    particles = agent.estimator_.particles_
    reso = agent.dp.reso
    pose_min = agent.pose_min
    index_nums = agent.dp.index_nums
    indices =
        [to_index(particle.pose_, pose_min, reso, index_nums) for particle in particles]
    value_function = agent.dp.value_function_
    agent.current_value_ =
        sum([value_function[index...] for index in indices]) / length(indices)

    max_val = -1e100
    a = nothing
    for action in agent.dp.actions
        val = evaluation(agent, action, indices)
        if val > max_val
            max_val = val
            a = copy(action)
        end
    end
    return a
end
