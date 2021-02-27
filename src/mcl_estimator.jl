using JuliaProbo

mutable struct Particle
    pose_::Vector{Float64}
    function Particle(pose::Vector{Float64})
        @assert length(pose) == 3
        new(copy(pose))
    end
end

function motion_update(p::Particle, v::Float64, ω::Float64, dt::Float64, mv::MvNormal{Float64})
    noises = rand(mv)
    noised_v = v + noises[1] * sqrt(abs(v)/dt) + noises[2] * sqrt(abs(ω)/dt)
    noised_ω = ω + noises[3] * sqrt(abs(v)/dt) + noises[4] * sqrt(abs(ω)/dt)
    p.pose_ = state_transition(p.pose_, noised_v, noised_ω, dt)
end

mutable struct Mcl <: AbstractEstimator
    particles_::Vector{Particle}
    motion_noise_rate_pdf::MvNormal{Float64}
    function Mcl(initial_pose::Vector{Float64},
                 num::Int64,
                 motion_noise_stds::Dict{String, Float64})
        v = motion_noise_stds
        cov = Diagonal([v["vv"]^2, v["vω"]^2, v["ωv"]^2, v["ωω"]^2])
        
        new([Particle(initial_pose) for i in 1:num],
            MvNormal([0.0, 0.0, 0.0, 0.0], cov))
    end
end

function motion_update(mcl::Mcl, v::Float64, ω::Float64, dt::Float64)
    N = length(mcl.particles_)
    for i in 1:N
        motion_update(mcl.particles_[i], v, ω, dt, mcl.motion_noise_rate_pdf)
    end
end

function draw(mcl::Mcl, p::Plot{T}) where T
    xs = [p.pose_[1] for p = mcl.particles_]
    ys = [p.pose_[2] for p = mcl.particles_]
    vxs = [cos(p.pose_[3]) for p = mcl.particles_]
    vys = [sin(p.pose_[3]) for p = mcl.particles_]
    p = quiver!(xs, ys, quiver=(vxs, vys), color="blue", alpha=0.5)
end
