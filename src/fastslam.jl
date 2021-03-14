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

mutable struct FastSlam <: AbstractMcl
    particles_::Vector{MapParticle}
    motion_noise_rate_pdf::MvNormal{Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    ml_::MapParticle
    pose_::Vector{Float64}
    function FastSlam(
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

function observation_update(
    mcl::FastSlam,
    observation::Vector{Vector{Float64}},
    envmap::Map;
    resample = true,
)
    # currently same as the one for mcl::Mcl
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

function draw(mcl::FastSlam, p::Plot{T}) where {T}
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
    # draw the belief map of maximum likelihood particle
    draw(mcl.ml_.map_, p)
end
