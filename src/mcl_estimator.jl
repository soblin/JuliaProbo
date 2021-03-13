mutable struct Particle <: AbstractParticle
    pose_::Vector{Float64}
    weight_::Float64
    function Particle(pose::Vector{Float64}, weight::Float64)
        @assert length(pose) == 3
        new(copy(pose), weight)
    end
end

function Base.copy(p::Particle)
    p_ = Particle(copy(p.pose_), p.weight_)
    return p_
end

function motion_update(
    p::AbstractParticle,
    v::Float64,
    ω::Float64,
    dt::Float64,
    mv::MvNormal{Float64},
)
    noises = rand(mv)
    noised_v = v + noises[1] * sqrt(abs(v) / dt) + noises[2] * sqrt(abs(ω) / dt)
    noised_ω = ω + noises[3] * sqrt(abs(v) / dt) + noises[4] * sqrt(abs(ω) / dt)
    p.pose_ = state_transition(p.pose_, noised_v, noised_ω, dt)
end

function observation_update(
    p::AbstractParticle,
    observation::Vector{Vector{Float64}},
    envmap::Map,
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    N = size(observation)[1]
    for i = 1:N
        obsv = observation[i]
        obs_pos = obsv[1:2]
        obs_id = convert(Int64, obsv[3])

        pos_on_map = envmap.landmarks_[obs_id+1].pos_
        particle_suggest_pos = observation_function(p.pose_, pos_on_map)

        distance_dev = distance_dev_rate * particle_suggest_pos[1]
        cov = [distance_dev^2 0.0; 0.0 direction_dev^2]
        p.weight_ *= pdf(MvNormal(particle_suggest_pos, cov), obs_pos)
    end
end

mutable struct Mcl <: AbstractMcl
    particles_::Vector{Particle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::Particle
    pose_::Vector{Float64}
    function Mcl(
        initial_pose::Vector{Float64},
        num::Int64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
        distance_dev_rate = 0.14,
        direction_dev = 0.05;
        glob = false, # GlobalMcl
        xlim = [-5.0, 5.0],
        ylim = [-5.0, 5.0],
    )
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        if glob
            pose_distrib = PoseUniform(xlim, ylim)
            initial_pose = uniform(pose_distrib)
            new(
                [Particle(uniform(pose_distrib), 1.0 / num) for i = 1:num],
                MvNormal([0.0, 0.0, 0.0, 0.0], cov),
                distance_dev_rate,
                direction_dev,
                Particle(initial_pose, 0.0),
                copy(initial_pose),
            )
        else
            new(
                [Particle(initial_pose, 1.0 / num) for i = 1:num],
                MvNormal([0.0, 0.0, 0.0, 0.0], cov),
                distance_dev_rate,
                direction_dev,
                Particle(initial_pose, 0.0),
                copy(initial_pose),
            )
        end
    end
end

mutable struct ResetMcl <: AbstractResetMcl
    particles_::Vector{Particle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::Particle
    pose_::Vector{Float64}
    αs_::Dict{Int64,Vector{Float64}}
    α_threshold::Float64
    reset_distrib::PoseUniform
    function ResetMcl(
        initial_pose::Vector{Float64},
        num::Int64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
        distance_dev_rate = 0.14,
        direction_dev = 0.05;
        α_threshold = 0.001,
        xlim = [-5.0, 5.0],
        ylim = [-5.0, 5.0],
    )
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        new(
            [Particle(initial_pose, 1.0 / num) for i = 1:num],
            MvNormal([0.0, 0.0, 0.0, 0.0], cov),
            distance_dev_rate,
            direction_dev,
            Particle(initial_pose, 0.0),
            copy(initial_pose),
            Dict{Int64,Vector{Float64}}(),
            α_threshold,
            PoseUniform(xlim, ylim),
        )
    end
end

mutable struct AMcl <: AbstractResetMcl
    particles_::Vector{Particle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::Particle
    pose_::Vector{Float64}
    reset_distrib::PoseUniform
    amcl_params::Dict{String,Float64}
    fast_α_term::Float64
    slow_α_term::Float64
    function AMcl(
        initial_pose::Vector{Float64},
        num::Int64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
        distance_dev_rate = 0.14,
        direction_dev = 0.05;
        xlim = [-5.0, 5.0],
        ylim = [-5.0, 5.0],
        amcl_params = Dict("slow" => 0.001, "fast" => 0.1, "v" => 3.0),
    )
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        new(
            [Particle(initial_pose, 1.0 / num) for i = 1:num],
            MvNormal([0.0, 0.0, 0.0, 0.0], cov),
            distance_dev_rate,
            direction_dev,
            Particle(initial_pose, 0.0),
            copy(initial_pose),
            PoseUniform(xlim, ylim),
            amcl_params,
            1.0,
            1.0,
        )
    end
end

function set_ml(mcl::AbstractMcl)
    ind = findmax([p.weight_ for p in mcl.particles_])[2]
    mcl.ml_ = copy(mcl.particles_[ind])
    mcl.pose_ = copy(mcl.ml_.pose_)
end

function motion_update(mcl::AbstractMcl, v::Float64, ω::Float64, dt::Float64)
    N = length(mcl.particles_)
    for i = 1:N
        motion_update(mcl.particles_[i], v, ω, dt, mcl.motion_noise_rate_pdf)
    end
end

function observation_update(
    mcl::Mcl,
    observation::Vector{Vector{Float64}},
    envmap::Map;
    resample = true,
)
    N = length(mcl.particles_)
    for i = 1:N
        observation_update(
            mcl.particles_[i],
            observation,
            envmap,
            mcl.distance_dev_rate,
            mcl.direction_dev,
        )
    end
    set_ml(mcl)
    if resample
        resampling(mcl)
    end
end

function observation_update(
    mcl::ResetMcl,
    observation::Vector{Vector{Float64}},
    envmap::Map;
    resample = true,
    sensor_reset = true,
)
    N = length(mcl.particles_)
    for i = 1:N
        observation_update(
            mcl.particles_[i],
            observation,
            envmap,
            mcl.distance_dev_rate,
            mcl.direction_dev,
        )
    end

    set_ml(mcl)
    α = sum([p.weight_ for p in mcl.particles_])
    if α < mcl.α_threshold
        if sensor_reset
            sensor_resetting(mcl, observation, envmap)
        end
    else
        if resample
            resampling(mcl)
        end
    end
end

function observation_update(mcl::AMcl, observation::Vector{Vector{Float64}}, envmap::Map)
    N = length(mcl.particles_)
    for i = 1:N
        observation_update(
            mcl.particles_[i],
            observation,
            envmap,
            mcl.distance_dev_rate,
            mcl.direction_dev,
        )
    end
    set_ml(mcl)
    adaptive_resetting(mcl, observation, envmap)
end

function resampling(mcl::AbstractMcl)
    ws = cumsum([p.weight_ for p in mcl.particles_])

    if ws[end] < 1e-100
        ws = [e + 1e-100 for e in ws]
    end

    step = ws[end] / length(ws)
    r = rand(Uniform(0.0, step))
    cur_pos = 1
    ps = []

    while length(ps) < length(mcl.particles_)
        if r < ws[cur_pos]
            push!(ps, copy(mcl.particles_[cur_pos]))
            r += step
        else
            cur_pos += 1
        end
    end

    mcl.particles_ = copy(ps)
    for i = 1:length(ps)
        mcl.particles_[i].weight_ = 1.0 / length(ps)
    end
end

function draw(mcl::AbstractMcl, p::Plot{T}) where {T}
    xs = [p.pose_[1] for p in mcl.particles_]
    ys = [p.pose_[2] for p in mcl.particles_]
    vxs =
        [cos(p.pose_[3]) * 0.5 * p.weight_ * length(mcl.particles_) for p in mcl.particles_]
    vys =
        [sin(p.pose_[3]) * 0.5 * p.weight_ * length(mcl.particles_) for p in mcl.particles_]
    p = quiver!(xs, ys, quiver = (vxs, vys), color = "blue", alpha = 0.5)
    pose = mcl.pose_
    annota = "($(round(pose[1], sigdigits=3)), $(round(pose[2], sigdigits=3)), $(round(pose[3], sigdigits=3)))"
    p = annotate!(pose[1] + 1.0, pose[2] + 1.0, text(annota, 10))
end

function sensor_resetting(
    mcl::AbstractResetMcl,
    observation::Vector{Vector{Float64}},
    envmap::Map,
)
    d_obs = findmin([obsv[1] for obsv in observation])
    nearest_ind = d_obs[2]
    d_obs = d_obs[1]
    values = observation[nearest_ind][1:2]
    landmark_id = convert(Int64, observation[nearest_ind][3] + 1)
    @assert values[1] == d_obs
    ψ_obs = values[2]

    N = length(mcl.particles_)
    for i = 1:N
        sensor_resetting_draw(
            mcl,
            mcl.particles_[i],
            envmap[landmark_id].pos_,
            d_obs,
            ψ_obs,
        )
    end
end

function sensor_resetting_draw(
    mcl::AbstractResetMcl,
    p::Particle,
    landmark_pos::Vector{Float64},
    d_obs::Float64,
    ϕ_obs::Float64,
)
    ψ = (rand() - 0.5) * (2 * pi) # ∈ [-π, π]
    d = rand(Normal(d_obs, (mcl.distance_dev_rate * d_obs)^2))
    p.pose_[1] = landmark_pos[1] + d * cos(ψ)
    p.pose_[2] = landmark_pos[2] + d * sin(ψ)

    ϕ = rand(Normal(ϕ_obs, mcl.direction_dev^2))
    p.pose_[3] = atan(landmark_pos[2] - p.pose_[2], landmark_pos[1] - p.pose_[1]) - ϕ
    p.weight_ = 1.0 / length(mcl.particles_)
end

function adaptive_resetting(mcl::AMcl, observation::Vector{Vector{Float64}}, envmap::Map)
    if length(observation) == 0
        return
    end

    amcl_params = mcl.amcl_params
    α = sum([p.weight_ for p in mcl.particles_])
    mcl.slow_α_term += amcl_params["slow"] * (α - mcl.slow_α_term)
    mcl.fast_α_term += amcl_params["fast"] * (α - mcl.fast_α_term)
    sl_num = 1.0 - amcl_params["v"] * mcl.fast_α_term / mcl.slow_α_term
    sl_num = length(mcl.particles_) * maximum([0.0, sl_num])

    resampling(mcl)
    nearest_obs = findmin([obsv[1] for obsv in observation])
    ind = nearest_obs[2]
    obs_d, obs_ψ = observation[ind][1], observation[ind][2]
    landmark_id = convert(Int64, observation[ind][3] + 1)

    N = length(mcl.particles_)
    for n = 1:convert(Int64, floor(sl_num))
        ind = rand(1:N)
        p = mcl.particles_[ind]
        sensor_resetting_draw(mcl, p, envmap[landmark_id].pos_, obs_d, obs_ψ)
    end
end

mutable struct KldMcl <: AbstractMcl
    particles_::Vector{Particle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::Particle
    pose_::Vector{Float64}
    reso::Vector{Float64}
    ϵ::Float64
    δ::Float64
    maxnum::Int64
    binnum_::Int64
    function KldMcl(
        initial_pose::Vector{Float64},
        maxnum::Int64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
        distance_dev_rate = 0.14,
        direction_dev = 0.05,
        reso = [0.2, 0.2, pi / 18],
        ϵ = 0.1,
        δ = 0.01,
    )
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])

        new(
            [Particle(initial_pose, 1.0)],
            MvNormal([0.0, 0.0, 0.0, 0.0], cov),
            distance_dev_rate,
            direction_dev,
            Particle(initial_pose, 1.0),
            copy(initial_pose),
            reso,
            ϵ,
            δ,
            maxnum,
            0,
        )
    end
end

function motion_update(mcl::KldMcl, v::Float64, ω::Float64, dt::Float64)
    ws = [p.weight_ for p in mcl.particles_]

    if sum(ws) < 1e-100
        ws = [e + 1e-100 for e in ws]
    end

    new_particles = Vector{Particle}(undef, 0)
    bins = Set{Vector{Int}}()
    for cnt = 1:mcl.maxnum
        p = sample(mcl.particles_, Weights(ws))
        motion_update(p, v, ω, dt, mcl.motion_noise_rate_pdf)
        bin = [convert(Int64, ceil(p.pose_[i] / mcl.reso[i])) for i = 1:length(p.pose_)]
        # local variables must be copied in Julia??
        push!(bins, copy(bin))
        push!(new_particles, copy(p))

        mcl.binnum_ = (length(bins) > 1) ? length(bins) : 2
        if length(new_particles) >
           ceil(chisqinvcdf(mcl.binnum_ - 1, 1.0 - mcl.δ) / (2 * mcl.ϵ))
            break
        end
    end

    mcl.particles_ = new_particles
    N = length(new_particles)
    for i = 1:N
        mcl.particles_[i].weight_ = 1.0 / N
    end
end

function observation_update(
    mcl::KldMcl,
    observation::Vector{Vector{Float64}},
    envmap::Map;
    resample = false,
)
    N = length(mcl.particles_)
    for i = 1:N
        observation_update(
            mcl.particles_[i],
            observation,
            envmap,
            mcl.distance_dev_rate,
            mcl.direction_dev,
        )
    end
    set_ml(mcl)
end

function draw(mcl::KldMcl, p::Plot{T}) where {T}
    xs = [p.pose_[1] for p in mcl.particles_]
    ys = [p.pose_[2] for p in mcl.particles_]
    vxs =
        [cos(p.pose_[3]) * 0.5 * p.weight_ * length(mcl.particles_) for p in mcl.particles_]
    vys =
        [sin(p.pose_[3]) * 0.5 * p.weight_ * length(mcl.particles_) for p in mcl.particles_]
    p = quiver!(xs, ys, quiver = (vxs, vys), color = "blue", alpha = 0.5)
    pose = mcl.pose_
    annota = "($(round(pose[1], sigdigits=3)), $(round(pose[2], sigdigits=3)), $(round(pose[3], sigdigits=3)), particle: $(length(mcl.particles_)), bins: $(mcl.binnum_))"
    p = annotate!(pose[1] + 1.0, pose[2] + 1.0, text(annota, 10))
end
