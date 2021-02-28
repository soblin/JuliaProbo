using JuliaProbo

mutable struct KalmanFilter <: AbstractEstimator
    belief_::MvNormal{Float64}
    motion_noise_stds::Dict{String, Float64}
    distance_dev_rate::Float64
    direction_dev::Float64
    pose_::Vector{Float64}
    map_::Map
    function KalmanFilter(envmap::Map,
                          initial_pose::Vector{Float64},
                          motion_noise_stds=
                          Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
                          distance_dev_rate=0.14,
                          direction_dev=0.05)
        @assert length(initial_pose) == 3
        cov = Diagonal([1e-10, 1e-10, 1e-10])
        new(MvNormal(initial_pose, cov),
            motion_noise_stds,
            distance_dev_rate,
            direction_dev,
            copy(initial_pose),
            envmap)
    end
end

function motion_update(kf::KalmanFilter, v::Float64, ω::Float64, dt::Float64)
    if abs(ω) < 1e-5
        ω = 1e-5
    end
    M = matM(v, ω, dt, kf.motion_noise_stds)
    A = matA(v, ω, dt, mean(kf.belief_)[3])
    F = matF(v, ω, dt, mean(kf.belief_)[3])
    kf.belief_.Σ = F * kf.belief_.Σ * transpose(F) + A * M * transpose(A)
    kf.belief_.μ = state_transition(mean(kf.belief_), v, ω, dt)
    kf.pose_ = mean(kf.belief_)
end

function observation_update(kf::KalmanFilter, observation::Vector{Vector{Float64}})
    return
end
