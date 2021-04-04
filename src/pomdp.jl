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

mutable struct BeliefDP
    pose_min::Vector{Float64}
    pose_max::Vector{Float64}
    reso::Vector{Float64}
    goal::Goal
    index_nums::Vector{Int64}
    indices::Vector{Tuple{Int64,Int64,Int64,Int64}}
    value_function_::AbstractArray{Float64,4}
    final_state_flags_::AbstractArray{Float64,4}
    policy_::AbstractArray{Float64,5}
    actions::Set{Vector{Float64}}
    state_transition_probs::Dict{
        Tuple{Float64,Float64,Int64}, # key is (v::Float64, ω::Float64, θ_index::Int64,)
        Vector{Tuple{Vector{Int64},Float64}}, # value is vector of ([tran_x_id, tran_y_id, tran_z_id], prob,)
    }
    depth::AbstractArray{Float64,2}
    dt::Float64
    puddle_coeff::Float64
    dev_borders::Vector{Float64}
    dev_borders_side::Vector{Float64}
    motion_sigma_transition_probs::Dict{
        Tuple{Int64,Vector{Float64}},
        Vector{Tuple{Int64,Float64}},
    }
    obs_sigma_transition_probs::Dict{Tuple{Int64,Int64,Int64,Int64},Tuple{Int64,Float64}}
    depths::AbstractArray{Float64,3}
end

# constructor
function BeliefDP(
    reso::Vector{Float64},
    goal::Goal;
    lowerleft = [-4.0, -4.0],
    upperright = [4.0, 4.0],
    dt = 0.1,
    puddle_coeff = 100.0,
    dev_borders = [0.1, 0.2, 0.4, 0.8],
)
    pose_min = vcat(lowerleft, [0.0])
    pose_max = vcat(upperright, [2pi])

    index_nums = [convert(Int64, round((pose_max[i] - pose_min[i]) / reso[i])) for i = 1:3]
    push!(index_nums, length(dev_borders) + 1)
    v = zeros(Float64, index_nums...)
    f = zeros(Float64, index_nums...)
    indices = Vector{Tuple{Int64,Int64,Int64,Int64}}(undef, 0)
    policy_ = zeros(Float64, index_nums..., 2)
    actions = Set{Vector{Float64}}()
    state_transition_probs =
        Dict{Tuple{Float64,Float64,Int64},Vector{Tuple{Vector{Int64},Float64}}}()
    depth = zeros(Float64, index_nums[1], index_nums[2])
    dev_borders_side = [dev_borders[1] / 10, dev_borders..., dev_borders[end] * 10]
    motion_sigma_transition_probs =
        Dict{Tuple{Int64,Vector{Float64}},Vector{Tuple{Int64,Float64}}}()
    obs_sigma_transition_probs = Dict{Tuple{Int64,Int64,Int64,Int64},Tuple{Int64,Float64}}()
    depths = zeros(Float64, index_nums[1], index_nums[2], index_nums[3])

    return BeliefDP(
        pose_min,
        pose_max,
        reso,
        goal,
        index_nums,
        indices,
        v,
        f,
        policy_,
        actions,
        state_transition_probs,
        depth,
        dt,
        puddle_coeff,
        dev_borders,
        dev_borders_side,
        motion_sigma_transition_probs,
        obs_sigma_transition_probs,
        depths,
    )
end

function set_belief_final_state(
    reso::Vector{Float64},
    goal::Goal,
    pose_min::Vector{Float64},
    index::Vector{Int64},
)
    x_min, y_min, θ_min = pose_min .+ reso .* (index[1:3] .- 1)
    x_max, y_max, θ_max = pose_min .+ reso .* (index[1:3])

    corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    ret = convert(Float64, all([inside(goal, corner) for corner in corners]))
    return ret * convert(Float64, index[4] == 1)
end

function init_value(agent::BeliefDP)
    index_nums = agent.index_nums
    goal = agent.goal
    reso = agent.reso
    pose_min = agent.pose_min
    v = agent.value_function_
    f = agent.final_state_flags_
    for id1 = 1:index_nums[1]
        for id2 = 1:index_nums[2]
            for id3 = 1:index_nums[3]
                for id4 = 1:index_nums[4]
                    index = (id1, id2, id3, id4)
                    val = set_belief_final_state(reso, goal, pose_min, [index...])
                    @inbounds f[index...] = val
                    @inbounds v[index...] = (val == 1.0) ? goal.value : (-100.0)
                    # this line is problematic if we use @distributed
                    push!(agent.indices, index)
                end
            end
        end
    end
end

function init_policy(agent::BeliefDP)
    indices = agent.indices
    reso = agent.reso
    pose_min = agent.pose_min
    goal = agent.goal

    for index in indices
        center = pose_min .+ reso .* ([index[1:3]...] * 1.0 .- 0.5)
        agent.policy_[index..., :] = initial_policy(center, goal)
    end
end

function init_state_transition_probs(agent::BeliefDP; sampling_num = 100)
    dt = agent.dt
    reso = agent.reso
    indices = agent.indices
    policy = agent.policy_
    actions = Set{Vector{Float64}}()
    index_nums = agent.index_nums
    pose_min = agent.pose_min
    actions = agent.actions
    for index in indices
        push!(actions, policy[index..., :])
    end

    dxs = collect(range(0.001, reso[1] * 0.999, length = sampling_num))
    dys = collect(range(0.001, reso[2] * 0.999, length = sampling_num))
    dθs = collect(range(0.001, reso[3] * 0.999, length = sampling_num))
    # key is (v::Float64, ω::Float64, θ_index::Int64,)
    # value is vector of ([tran_x_id, tran_y_id, tran_z_id], prob,)
    transition_probs = agent.state_transition_probs

    for a in actions
        for i_θ = 1:index_nums[3]
            transitions = Vector{Vector{Int64}}(undef, 0)
            for dx in dxs
                for dy in dys
                    for dθ in dθs
                        before = [dx, dy, dθ + (i_θ - 1) * reso[3]] .+ pose_min
                        before_index = [1, 1, i_θ]
                        after = state_transition(before, a[1], a[2], dt)
                        after_index_ = (after - pose_min) ./ reso
                        after_index = map(x -> convert(Int64, floor(x) + 1), after_index_)
                        push!(transitions, after_index - before_index)
                    end
                end
            end
            elems = unique(transitions)
            counts = [count(x -> x == elem, transitions) for elem in elems]
            probs = [c / (sampling_num)^3 for c in counts]
            transition_probs[(a[1], a[2], i_θ)] = [iter for iter in zip(elems, probs)]
        end
    end
end

function init_depth(agent::BeliefDP, world::PuddleWorld; sampling_num = 100)
    reso = agent.reso
    dxs = collect(range(0, reso[1], length = sampling_num))
    dys = collect(range(0, reso[2], length = sampling_num))
    index_nums = agent.index_nums
    pose_min = agent.pose_min
    puddles = world.puddles_
    depth = agent.depth

    for x = 0:index_nums[1]-1
        for y = 0:index_nums[2]-1
            for dx in dxs
                for dy in dys
                    pose = pose_min .+ (reso .* [x, y, 0]) .+ [dx, dy, 0]
                    for puddle in puddles
                        depth[x+1, y+1] +=
                            (puddle.depth * (convert(Float64, inside(puddle, pose))))
                    end
                end
            end
            depth[x+1, y+1] /= (sampling_num)^2
        end
    end
end

function correct_index(agent::BeliefDP, index_::Vector{Int64})
    index = copy(index_)
    index_nums = agent.index_nums
    reso = agent.reso
    while index[3] < 1
        index[3] += index_nums[3]
    end
    while index[3] > index_nums[3]
        index[3] -= index_nums[3]
    end
    out_reward = 0.0
    for i in [1, 2]
        if index[i] < 1
            index[i] = 1
            out_reward = -1e100
        elseif index[i] > index_nums[i]
            index[i] = index_nums[i]
            out_reward = -1e100
        end
    end
    return index, out_reward
end

function action_value(
    agent::BeliefDP,
    action::Vector{Float64},
    index::Vector{Int64},
    value_function::AbstractArray{Float64,4},
)
    value = 0.0
    dt = agent.dt
    transition = agent.state_transition_probs[(action[1], action[2], index[3])]
    depth = agent.depth
    puddle_coeff = agent.puddle_coeff
    motion_transition = agent.motion_sigma_transition_probs
    obs_transition = agent.obs_sigma_transition_probs

    for (delta, prob) in transition
        after, out_reward = correct_index(agent, index[1:3] + [delta...])
        push!(after, 1)
        reward = -dt * depth[after[1:2]...] * puddle_coeff - dt + out_reward
        if haskey(motion_transition, (index[4], action))
            for (σ_after, σ_prob) in motion_transition[(index[4], action)]
                after[4] = σ_after
                σ_obs, σ_obs_prob = obs_transition[(after...,)]
                value +=
                    (value_function[after[1:3]..., σ_obs] + reward) *
                    σ_prob *
                    prob *
                    σ_obs_prob
            end
        end
    end
    return value
end

function value_iteration_sweep(agent::BeliefDP; γ = 1.0)
    max_Δ = 0.0
    indices = agent.indices
    final_state_flags = agent.final_state_flags_
    value_function = copy(agent.value_function_)
    for index in indices
        if final_state_flags[index...] == 0.0
            max_a = nothing
            max_q = -1e100
            for action in agent.actions
                q = action_value(agent, action, [index...], value_function)
                if q > max_q
                    max_a = copy(action)
                    max_q = q
                end
            end
            Δ = abs(value_function[index...] - max_q)
            max_Δ = max(Δ, max_Δ)
            agent.value_function_[index...] = max_q
            agent.policy_[index..., :] = max_a
        end
    end
    return max_Δ
end

function cov_to_index(dev_borders::Vector{Float64}, cov_::Matrix{Float64})
    σ = det(cov_)^(1.0 / 6)
    for (i, elem) in enumerate(dev_borders)
        if σ < elem
            return i
        end
    end
    return length(dev_borders)
end

function cov_to_index(agent::BeliefDP, cov_::Matrix{Float64})
    σ = det(cov_)^(1.0 / 6)
    for (i, elem) in enumerate(agent.dev_borders)
        if σ < elem
            return i
        end
    end
    return length(agent.dev_borders)
end

function calc_motion_sigma_transition_probs(
    agent::BeliefDP,
    min_σ::Float64,
    max_σ::Float64,
    action::Vector{Float64};
    sampling_num = 100,
)
    dt = agent.dt
    v, ω = action[1], action[2]
    if abs(ω) < 1e-5
        ω = 1e-5
    end
    F = matF(v, ω, dt, 0.0)
    M = matM(v, ω, dt, Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2))
    A = matA(v, ω, dt, 0.0)
    indices = Dict{Int64,Float64}()
    for σ in range(min_σ, max_σ * 0.999, length = sampling_num)
        cov_ = σ * σ * F * transpose(F) + A * M * transpose(A)
        index_after = cov_to_index(agent.dev_borders, cov_)
        if !haskey(indices, index_after)
            indices[index_after] = 1
        else
            indices[index_after] += 1
        end
    end
    for (k, v) in indices
        indices[k] /= sampling_num
    end
    return [(k, v) for (k, v) in indices]
end

function init_motion_sigma_transition_probs(agent::BeliefDP)
    probs = agent.motion_sigma_transition_probs
    dev_borders_side = agent.dev_borders_side
    for a in agent.actions
        for i = 1:length(agent.dev_borders)+1
            probs[(i, a)] = calc_motion_sigma_transition_probs(
                agent,
                dev_borders_side[i],
                dev_borders_side[i+1],
                a,
            )
        end
    end
end

function observation_update(
    lm_id::Int64,
    S::Matrix{Float64},
    camera::IdealCamera,
    pose::Vector{Float64},
)
    distance_dev_rate = 0.14
    direction_dev = 0.05

    H = matH(pose, camera.landmarks_[lm_id+1].pos_)
    est_z = observation_function(pose, camera.landmarks_[lm_id+1].pos_)
    Q = matQ(distance_dev_rate * est_z[1], direction_dev)
    K = S * transpose(H) * (inv(Q + H * S * transpose(H)))
    return (Matrix(1.0I, 3, 3) - K * H) * S
end

function init_obs_sigma_transition_probs(agent::BeliefDP, camera::IdealCamera)
    sigma_transition = agent.obs_sigma_transition_probs
    pose_min = agent.pose_min
    reso = agent.reso
    dev_borders_side = agent.dev_borders_side
    for index in agent.indices
        pose = pose_min .+ reso .* ([index[1:3]...] * 1.0 .- 0.5)
        σ = (dev_borders_side[index[4]] + dev_borders_side[index[4]+1]) / 2.0
        S = Matrix(σ^2 * I, 3, 3)
        for d in observations(camera, pose)
            lm_id = convert(Int64, d[3])
            S = observation_update(lm_id, S, camera, pose)
        end
        sigma_transition[index] = (cov_to_index(agent, S), 1.0)
    end
end

function init_expected_depths(agent::BeliefDP, world::PuddleWorld; sampling_num = 100)
    puddles = world.puddles_
    index_nums = agent.index_nums
    reso = agent.reso
    pose_min = agent.pose_min
    dev_borders_side = agent.dev_borders_side
    depths = agent.depths
    for id1 in index_nums[1]
        for id2 in index_nums[2]
            for id3 in index_nums[4]
                index = [id1, id2, id3]
                pose = pose_min[1:2] .+ reso[1:2] .* (index[1:2] .- 0.5)
                σ = (dev_borders_side[id3] + dev_borders_side[id3+1]) / 2.0
                belief = MvNormal(pose, Matrix(σ^2 * I, 2, 2))
                depth_sum = 0.0
                samples = rand(belief, sampling_num)
                for i_pos = 1:sampling_num
                    pos = samples[:, i_pos]
                    depth_sum += sum([
                        puddle.depth * convert(Float64, inside(puddle, pos)) for
                        puddle in puddles
                    ])
                end
                depths[index...] = depth_sum / sampling_num
            end
        end
    end
end

mutable struct AMDPPolicyAgent <: AbstractMDPAgent
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
    policy_data::AbstractArray{Float64,5}
    dev_borders::Vector{Float64}
end

function AMDPPolicyAgent(
    v::Float64,
    ω::Float64,
    dt::Float64,
    estimator::AbstractEstimator,
    goal::Goal,
    reso::Vector{Float64};
    lowerleft = [-4.0, -4.0],
    upperright = [4.0, 4.0],
    puddle_coeff = 100,
    dev_borders = [0.1, 0.2, 0.4, 0.8],
)
    pose_min = vcat(lowerleft, [0.0])
    pose_max = vcat(upperright, [2pi])
    index_nums = [convert(Int64, round((pose_max[i] - pose_min[i]) / reso[i])) for i = 1:3]
    push!(index_nums, length(dev_borders) + 1)
    policy_data = zeros(Float64, index_nums..., 2)
    return AMDPPolicyAgent(
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
        dev_borders,
    )
end

function init_policy(agent::AMDPPolicyAgent, fname = "policy.txt")
    policy_data = agent.policy_data
    fd = open(fname, "r")
    lines = readlines(fd)
    for line in lines
        tokens = split(line, " ")
        tokens = map(x -> parse(Float64, x), tokens)
        id1, id2, id3, id4 = map(x -> convert(Int64, x), tokens[1:4])
        v, ω = tokens[5], tokens[6]
        policy_data[id1, id2, id3, id4, :] = [v, ω]
    end
end

function policy(agent::AMDPPolicyAgent)
    reso = agent.reso
    pose = agent.estimator_.pose_
    pose_min = agent.pose_min
    index_nums = agent.index_nums
    # normalize index
    index = [convert(Int64, round((pose[i] - pose_min[i]) / reso[i])) for i = 1:3]
    while index[3] < 1
        index[3] += index_nums[3]
    end
    while index[3] > index_nums[3]
        index[3] -= index_nums[3]
    end
    for i in [1, 2]
        if index[i] < 1
            index[i] = 1
        end
        if index[i] > index_nums[i]
            index[i] = index_nums[i]
        end
    end
    belief_index = cov_to_index(agent.dev_borders, cov(agent.estimator_.belief_))
    push!(index, belief_index)
    a = agent.policy_data[index..., :]
    return a
end
