mutable struct EstimatedLandmark <: AbstractLandmark
    pos_::Vector{Float64}
    id::Int64
    cov_::Union{Matrix{Float64},Nothing}
    function EstimatedLandmark(id::Int64)
        # this cov_ initial value is for debugging
        new([0.0, 0.0], id, nothing)
    end
end

function Base.copy(lm::EstimatedLandmark)
    if lm.cov_ == nothing
        return EstimatedLandmark(copy(lm.pos_), lm.id, nothing)
    else
        return EstimatedLandmark(copy(lm.pos_), lm.id, copy(lm.cov_))
    end
end

function draw(mark::EstimatedLandmark, p::Plot{T}) where {T}
    if mark.cov_ == nothing
        return
    else
        p = scatter!(
            [mark.pos_[1]],
            [mark.pos_[2]],
            markershape = :star,
            markersize = 10,
            color = "blue",
        )
        p = annotate!(
            mark.pos_[1] + 0.5,
            mark.pos_[2] + 0.5,
            text("id: $(mark.id)", 10, :blue),
        )
        p = myellipse!(mark.pos_, mark.cov_[1:2, 1:2], n_std = 3, aspect_ratio = 1)
    end
end

mutable struct MapParticle <: AbstractParticle
    pose_::Vector{Float64}
    weight_::Float64
    map_::Map
    function MapParticle(init_pose::Vector{Float64}, weight::Float64, landmark_num::Int64)
        envmap = Map()
        for i = 1:landmark_num
            push!(envmap, EstimatedLandmark(i - 1))
        end
        new(copy(init_pose), weight, envmap)
    end
end

function Base.copy(p::MapParticle)
    p_ = MapParticle(copy(p.pose_), p.weight_, 0)
    p_.map_ = copy(p.map_)
    return p_
end

function motion_update2(
    particle::MapParticle,
    v::Float64,
    ω::Float64,
    dt::Float64,
    motion_noise_stds::Dict{String,Float64},
    observation::Vector{Vector{Float64}},
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    M = matM(v, ω, dt, motion_noise_stds)
    A = matA(v, ω, dt, particle.pose_[3])
    Rₜ = A * M * transpose(A)
    x̂ = state_transition(particle.pose_, v, ω, dt)

    N = size(observation)[1]
    for i = 1:N
        obsv = observation[i]
        z = obsv[1:2]
        idx = convert(Int64, obsv[3])
        x̂, Rₜ = gauss_for_drawing(
            x̂,
            Rₜ,
            z,
            particle.map_.landmarks_[idx+1],
            distance_dev_rate,
            direction_dev,
        )
    end

    Rₜ = (Rₜ + transpose(Rₜ)) / 2.0
    particle.pose_ = rand(MvNormal(x̂, Rₜ + Matrix(1.0e-10 * I, 3, 3)))
end

function drawing_params(
    x̂::Vector{Float64},
    lm::EstimatedLandmark,
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    d = hypot((x̂[1:2] - lm.pos_)...)
    Qzₜ̂ = matQ(distance_dev_rate * d, direction_dev)
    zₜ̂ = observation_function(x̂, lm.pos_)
    Hm = -matH(x̂, lm.pos_)[1:2, 1:2]
    Hxₜ = matH(x̂, lm.pos_)

    Qzₜ = Hm * lm.cov_ * transpose(Hm) + Qzₜ̂

    return zₜ̂, Qzₜ, Hxₜ
end

function gauss_for_drawing(
    x̂::Vector{Float64},
    Rₜ::Matrix{Float64},
    z::Vector{Float64},
    lm::EstimatedLandmark,
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    zₜ̂, Qzₜ, Hxₜ = drawing_params(x̂, lm, distance_dev_rate, direction_dev)
    K = Rₜ * transpose(Hxₜ) * inv(Qzₜ + Hxₜ * Rₜ * transpose(Hxₜ))

    return K * (z - zₜ̂) + x̂, (Matrix(1.0I, 3, 3) - K * Hxₜ) * Rₜ
end

function observation_update(
    particle::MapParticle,
    observation::Vector{Vector{Float64}},
    envmap::Map,
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    N = size(observation)[1]
    for i = 1:N
        obsv = observation[i]
        d, ϕ = obsv[1], obsv[2]
        lm_id = convert(Int64, obsv[3])
        landmark = particle.map_.landmarks_[lm_id+1]
        if landmark.cov_ == nothing
            init_landmark_estimation(
                particle,
                landmark,
                d,
                ϕ,
                distance_dev_rate,
                direction_dev,
            )
        else
            observation_update_landmark(
                particle,
                landmark,
                d,
                ϕ,
                distance_dev_rate,
                direction_dev,
            )
        end
    end
end

function init_landmark_estimation(
    particle::MapParticle,
    lm::EstimatedLandmark,
    d::Float64,
    ϕ::Float64,
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    θ = particle.pose_[3]
    lm.pos_ = particle.pose_[1:2] + d * [cos(ϕ + θ), sin(ϕ + θ)]
    H = matH(particle.pose_, lm.pos_)[1:2, 1:2]
    Q = matQ(distance_dev_rate * d, direction_dev)
    lm.cov_ = inv(transpose(H) * inv(Q) * H)
end

function observation_update_landmark(
    particle::MapParticle,
    lm::EstimatedLandmark,
    d::Float64,
    ϕ::Float64,
    distance_dev_rate::Float64,
    direction_dev::Float64,
)
    # estimated position [d, ϕ] from estimated position of landmark
    estm_z = observation_function(particle.pose_, lm.pos_)
    if estm_z[1] < 0.01
        # too close
        return
    end

    H = -matH(particle.pose_, lm.pos_)[1:2, 1:2]
    Q = matQ(distance_dev_rate * estm_z[1], direction_dev)
    K = lm.cov_ * transpose(H) * inv(Q + H * lm.cov_ * transpose(H))

    # update weight
    Q_z = H * lm.cov_ * transpose(H) + Q
    Q_z = (Q_z + transpose(Q_z)) / 2.0
    particle.weight_ *= pdf(MvNormal(estm_z, Q_z), [d, ϕ])

    # update landmark position esimate
    lm.pos_ = K * ([d, ϕ] - estm_z) + lm.pos_
    lm.cov_ = (Matrix(1.0I, 2, 2) - K * H) * lm.cov_
end

mutable struct FastSlam1 <: AbstractMcl
    particles_::Vector{MapParticle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::MapParticle
    pose_::Vector{Float64}
    function FastSlam1(
        init_pose::Vector{Float64},
        particle_num::Int64,
        landmark_num::Int64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
        distance_dev_rate = 0.14,
        direction_dev = 0.05,
    )
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        new(
            [
                MapParticle(init_pose, 1.0 / particle_num, landmark_num) for
                i = 1:particle_num
            ],
            MvNormal([0.0, 0.0, 0.0, 0.0], cov),
            distance_dev_rate,
            direction_dev,
            MapParticle(init_pose, 1.0, 0),
            init_pose,
        )
    end
end

mutable struct FastSlam2 <: AbstractMcl
    particles_::Vector{MapParticle}
    motion_noise_stds::Dict{String,Float64}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::MapParticle
    pose_::Vector{Float64}
    function FastSlam2(
        init_pose::Vector{Float64},
        particle_num::Int64,
        landmark_num::Int64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
        distance_dev_rate = 0.14,
        direction_dev = 0.05,
    )
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        new(
            [
                MapParticle(init_pose, 1.0 / particle_num, landmark_num) for
                i = 1:particle_num
            ],
            copy(v),
            MvNormal([0.0, 0.0, 0.0, 0.0], cov),
            distance_dev_rate,
            direction_dev,
            MapParticle(init_pose, 1.0, 0),
            init_pose,
        )
    end
end

function motion_update(slam::FastSlam1, v::Float64, ω::Float64, dt::Float64)
    N = length(slam.particles_)
    for i = 1:N
        motion_update(slam.particles_[i], v, ω, dt, slam.motion_noise_rate_pdf)
    end
end

function motion_update(
    slam::FastSlam2,
    v::Float64,
    ω::Float64,
    dt::Float64,
    observation::Vector{Vector{Float64}},
)
    not_first_observation = Vector{Vector{Float64}}(undef, 0)
    M = size(observation)[1]
    for i = 1:M
        obsv = observation[i]
        idx = convert(Int64, obsv[3])
        if slam.particles_[1].map_.landmarks_[idx+1].cov_ != nothing
            push!(not_first_observation, obsv)
        end
    end

    N = length(slam.particles_)
    if size(not_first_observation)[1] > 0
        for i = 1:N
            motion_update2(
                slam.particles_[i],
                v,
                ω,
                dt,
                slam.motion_noise_stds,
                not_first_observation,
                slam.distance_dev_rate,
                slam.direction_dev,
            )
            #motion_update(slam.particles_[i], v, ω, dt, slam.motion_noise_rate_pdf)
        end
    else
        for i = 1:N
            motion_update(slam.particles_[i], v, ω, dt, slam.motion_noise_rate_pdf)
        end
    end
end

function observation_update(
    slam::Union{FastSlam1,FastSlam2},
    observation::Vector{Vector{Float64}},
    envmap::Map;
    resample = true,
)
    # currently same as the one for mcl::Mcl
    N = length(slam.particles_)
    for i = 1:N
        observation_update(
            slam.particles_[i],
            observation,
            envmap,
            slam.distance_dev_rate,
            slam.direction_dev,
        )
    end
    set_ml(slam)
    if resample
        resampling(slam)
    end
end

function draw(slam::Union{FastSlam1,FastSlam2}, p::Plot{T}) where {T}
    xs = [p.pose_[1] for p in slam.particles_]
    ys = [p.pose_[2] for p in slam.particles_]
    vxs = [
        cos(p.pose_[3]) * 0.5 * p.weight_ * length(slam.particles_) for
        p in slam.particles_
    ]
    vys = [
        sin(p.pose_[3]) * 0.5 * p.weight_ * length(slam.particles_) for
        p in slam.particles_
    ]
    p = quiver!(xs, ys, quiver = (vxs, vys), color = "blue", alpha = 0.5)
    pose = slam.pose_
    annota = "($(round(pose[1], sigdigits=3)), $(round(pose[2], sigdigits=3)), $(round(pose[3], sigdigits=3)))"
    p = annotate!(pose[1] + 1.0, pose[2] + 1.0, text(annota, 10))
    # draw the belief map of maximum likelihood particle
    draw(slam.ml_.map_, p)
end
