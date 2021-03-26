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

mutable struct PolicyEvaluator
    pose_min::Vector{Float64}
    pose_max::Vector{Float64}
    reso::Vector{Float64}
    goal::Goal
    index_nums::Vector{Int64}
    indices::Vector{Tuple{Int64,Int64,Int64}}
    value_function_::AbstractArray{Float64,3}
    final_state_flags_::AbstractArray{Float64,3}
    policy_::AbstractArray{Float64,4}
    actions::Set{Vector{Float64}}
    state_transition_probs::Dict{
        Tuple{Float64,Float64,Int64}, # key is (v::Float64, ω::Float64, θ_index::Int64,)
        Vector{Tuple{Vector{Int64},Float64}}, # value is vector of ([tran_x_id, tran_y_id, tran_z_id], prob,)
    }
    depth::AbstractArray{Float64,2}
    dt::Float64
    puddle_coeff::Float64
end

function set_final_state(
    reso::Vector{Float64},
    goal::Goal,
    pose_min::Vector{Float64},
    index::Tuple{Int64,Int64,Int64},
)
    x_min, y_min, θ_min = pose_min .+ reso .* ([index...] .- 1)
    x_max, y_max, θ_max = pose_min .+ reso .* ([index...])

    corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return convert(Float64, all([inside(goal, corner) for corner in corners]))
end

function init_value(pe::PolicyEvaluator)
    index_nums = pe.index_nums
    goal = pe.goal
    reso = pe.reso
    pose_min = pe.pose_min
    v = pe.value_function_
    f = pe.final_state_flags_
    for id1 = 1:index_nums[1]
        for id2 = 1:index_nums[2]
            for id3 = 1:index_nums[3]
                index = (id1, id2, id3)
                val = set_final_state(reso, goal, pose_min, index)
                @inbounds f[index...] = val
                @inbounds v[index...] = (val == 1.0) ? goal.value : (-100.0)
                # this line is problematic if we use @distributed
                push!(pe.indices, (id1, id2, id3))
            end
        end
    end
end

function initial_policy(pose::Vector{Float64}, goal::Goal)
    x, y, θ = pose[1], pose[2], pose[3]
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
    [v, ω]
end

function init_policy(pe::PolicyEvaluator)
    indices = pe.indices
    reso = pe.reso
    pose_min = pe.pose_min
    goal = pe.goal

    for index in indices
        center = pose_min .+ reso .* ([index...] * 1.0 .- 0.5)
        pe.policy_[index..., :] = initial_policy(center, goal)
    end
end

function init_state_transition_probs(pe::PolicyEvaluator, dt::Float64, sampling_num::Int64)
    reso = pe.reso
    indices = pe.indices
    policy = pe.policy_
    actions = Set{Vector{Float64}}()
    index_nums = pe.index_nums
    pose_min = pe.pose_min
    actions = pe.actions
    for index in indices
        push!(actions, policy[index..., :])
    end

    dxs = collect(range(0.001, reso[1] * 0.999, length = sampling_num))
    dys = collect(range(0.001, reso[2] * 0.999, length = sampling_num))
    dθs = collect(range(0.001, reso[3] * 0.999, length = sampling_num))
    # key is (v::Float64, ω::Float64, θ_index::Int64,)
    # value is vector of ([tran_x_id, tran_y_id, tran_z_id], prob,)
    transition_probs = pe.state_transition_probs

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

function init_depth(pe::PolicyEvaluator, world::PuddleWorld, sampling_num::Int64)
    reso = pe.reso
    dxs = collect(range(0, reso[1], length = sampling_num))
    dys = collect(range(0, reso[2], length = sampling_num))
    index_nums = pe.index_nums
    pose_min = pe.pose_min
    puddles = world.puddles_
    depth = pe.depth

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

# constructor
function PolicyEvaluator(
    reso::Vector{Float64},
    goal::Goal;
    lowerleft = [-4.0, -4.0],
    upperright = [4.0, 4.0],
    dt = 0.1,
    puddle_coeff = 100.0,
)
    pose_min = vcat(lowerleft, [0.0])
    pose_max = vcat(upperright, [2pi])

    index_nums = [convert(Int64, round((pose_max[i] - pose_min[i]) / reso[i])) for i = 1:3]
    v = zeros(Float64, index_nums[1], index_nums[2], index_nums[3])
    f = zeros(Float64, index_nums[1], index_nums[2], index_nums[3])
    indices = Vector{Tuple{Int64,Int64,Int64}}(undef, 0)
    policy_ = zeros(Float64, index_nums..., 2)
    actions = Set{Vector{Float64}}()
    state_transition_probs =
        Dict{Tuple{Float64,Float64,Int64},Vector{Tuple{Vector{Int64},Float64}}}()
    depth = zeros(Float64, index_nums[1], index_nums[2])

    return PolicyEvaluator(
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
    )
end

function out_correction(pe::PolicyEvaluator, index_::Vector{Int64})
    index = copy(index_)
    θ_ind = (index[3] + pe.index_nums[3]) % (pe.index_nums[3])
    if θ_ind == 0
        θ_ind = pe.index_nums[3]
    end
    out_reward = 0.0
    for i = 1:2
        if index[i] < 1
            index[i] = 1
            out_reward = -1e100
        elseif index[i] > pe.index_nums[i]
            index[i] = pe.index_nums[i]
            out_reward = -1e100
        end
    end
    return (index[1], index[2], θ_ind), out_reward
end

function action_value(
    pe::PolicyEvaluator,
    action::Vector{Float64},
    index::Vector{Int64},
    value_function::AbstractArray{Float64,3};
    out_penalty = false,
)
    v, ω = action[1], action[2]
    value = 0.0
    state_transition_probs = pe.state_transition_probs
    for trans_probs in state_transition_probs[(v, ω, index[3])]
        trans_ind = trans_probs[1]
        prob = trans_probs[2]
        after_ = [index...] .+ trans_ind
        after, out_reward = out_correction(pe, after_)
        reward =
            -pe.dt * pe.depth[after[1], after[2]] * pe.puddle_coeff - pe.dt +
            out_reward * convert(Float64, out_penalty)
        value += (value_function[after...] + reward) * prob
    end
    return value
end

function policy_evaluation_sweep(pe::PolicyEvaluator)
    indices = pe.indices
    final_state_flags = pe.final_state_flags_
    value_function = copy(pe.value_function_)
    max_Δ = 0.0
    for index in indices
        if final_state_flags[index...] == 0.0
            q1 = value_function[index...]
            action = pe.policy_[index..., :]
            q2 = action_value(pe, action, [index...], value_function)
            pe.value_function_[index...] = q2
            Δ = abs(q2 - q1)
            if Δ > max_Δ
                max_Δ = Δ
            end
        end
    end
    return max_Δ
end

function value_iteration_sweep(pe::PolicyEvaluator)
    max_Δ = 0.0
    indices = pe.indices
    final_state_flags = pe.final_state_flags_
    value_function = copy(pe.value_function_)
    for index in indices
        if final_state_flags[index...] == 0.0
            max_a = nothing
            max_q = -1e100
            for action in pe.actions
                q = action_value(pe, action, [index...], value_function; out_penalty = true)
                if q > max_q
                    max_a = copy(action)
                    max_q = q
                end
            end
            Δ = abs(value_function[index...] - max_q)
            max_Δ = max(Δ, max_Δ)
            pe.value_function_[index...] = max_q
            pe.policy_[index..., :] = max_a
        end
    end
    return max_Δ
end
