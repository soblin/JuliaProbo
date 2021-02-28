using JuliaProbo

mutable struct Particle
    pose_::Vector{Float64}
    weight_::Float64
    function Particle(pose::Vector{Float64}, weight::Float64)
        @assert length(pose) == 3
        new(copy(pose), weight)
    end
end

function Base.copy(p::Particle)
    p_ = Particle(p.pose_, p.weight_)
    return p_
end

function motion_update(p::Particle, v::Float64, ω::Float64, dt::Float64, mv::MvNormal{Float64})
    noises = rand(mv)
    noised_v = v + noises[1] * sqrt(abs(v)/dt) + noises[2] * sqrt(abs(ω)/dt)
    noised_ω = ω + noises[3] * sqrt(abs(v)/dt) + noises[4] * sqrt(abs(ω)/dt)
    p.pose_ = state_transition(p.pose_, noised_v, noised_ω, dt)
end

function observation_update(p::Particle, observation::Vector{Vector{Float64}}, envmap::Map, distance_dev_rate::Float64, direction_dev::Float64)
    N = size(observation)[1]
    for i in 1:N
        obsv = observation[i]
        obs_pos = obsv[1:2]
        obs_id = convert(Int64, obsv[3])

        pos_on_map = envmap.landmarks_[obs_id+1].pos
        particle_suggest_pos = observation_function(p.pose_, pos_on_map)

        distance_dev = distance_dev_rate * particle_suggest_pos[1]
        cov = [distance_dev^2 0.0; 0.0 direction_dev^2]
        p.weight_ *= pdf(MvNormal(particle_suggest_pos, cov), obs_pos)
    end
end

mutable struct Mcl <: AbstractEstimator
    particles_::Vector{Particle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::Particle
    pose_::Vector{Float64}
    function Mcl(initial_pose::Vector{Float64},
                 num::Int64,
                 motion_noise_stds=
                 Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
                 distance_dev_rate=0.14,
                 direction_dev=0.05)
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        
        new([Particle(initial_pose, 1.0/num) for i in 1:num],
            MvNormal([0.0, 0.0, 0.0, 0.0], cov),
            distance_dev_rate,
            direction_dev,
            Particle(initial_pose, 0.0),
            copy(initial_pose))
    end
end

function set_ml(mcl::Mcl)
    i = findmax([p.weight_ for p in mcl.particles_])[2]
    mcl.ml_ = copy(mcl.particles_[i])
    mcl.pose_ = copy(mcl.ml_.pose_)
end

function motion_update(mcl::Mcl, v::Float64, ω::Float64, dt::Float64)
    N = length(mcl.particles_)
    for i in 1:N
        motion_update(mcl.particles_[i], v, ω, dt, mcl.motion_noise_rate_pdf)
    end
end

function observation_update(mcl::Mcl, observation::Vector{Vector{Float64}}, envmap::Map;
                            resample=false)
    N = length(mcl.particles_)
    for i in 1:N
        observation_update(mcl.particles_[i], observation, envmap, mcl.distance_dev_rate, mcl.direction_dev)
    end
    set_ml(mcl)
    if resample
        resampling(mcl)
    end
end

function resampling(mcl::Mcl)
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
    for i in 1:length(ps)
        mcl.particles_[i].weight_ = 1.0 / length(ps)
    end
end

function draw(mcl::Mcl, p::Plot{T}) where T
    xs = [p.pose_[1] for p = mcl.particles_]
    ys = [p.pose_[2] for p = mcl.particles_]
    vxs = [cos(p.pose_[3]) * 0.5 * p.weight_ * length(mcl.particles_) for p = mcl.particles_]
    vys = [sin(p.pose_[3]) * 0.5 * p.weight_ * length(mcl.particles_) for p = mcl.particles_]
    p = quiver!(xs, ys, quiver=(vxs, vys), color="blue", alpha=0.5)
    pose = mcl.pose_
    annota = "($(round(pose[1], sigdigits=3)), $(round(pose[2], sigdigits=3)), $(round(pose[3], sigdigits=3)))"
    p = annotate!(pose[1]+1.0, pose[2]+1.0, text(annota, 10))
end
