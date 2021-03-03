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
        new(MvNormal(initial_pose, Diagonal([1e-10, 1e-10, 1e-10])),
            motion_noise_stds,
            distance_dev_rate,
            direction_dev,
            copy(initial_pose),
            envmap)
    end
end

function matM(v::Float64, ω::Float64, dt::Float64, stds::Dict{String, Float64})
    return Diagonal([stds["vv"]^2 * abs(v) / dt + stds["vω"]^2 * abs(ω) / dt,
                     stds["ωv"]^2 * abs(v) / dt + stds["ωω"]^2 * abs(ω) / dt])
end

function matA(v::Float64, ω::Float64, dt::Float64, θ::Float64)
    st, ct = sin(θ), cos(θ)
    stw, ctw = sin(θ + ω * dt), cos(θ + ω * dt)
    a11 = (stw - st) / ω
    a12 = -v / (ω^2)*(stw - st) + v / ω * dt * ctw
    a21 = (-ctw + ct) / ω
    a22 = -v / (ω^2) * (-ctw + ct) + v / ω * dt * stw
    a31 = 0.0
    a32 = dt
    return [a11 a12;
            a21 a22;
            a31 a32]
end

function matF(v::Float64, ω::Float64, dt::Float64, θ::Float64)
    F = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    F[1, 3] = v / ω * ( cos(θ + ω * dt) - cos(θ))
    F[2, 3] = v / ω * ( sin(θ + ω * dt) - sin(θ))
    return F
end

function matH(pose::Vector{Float64}, landmark_pos::Vector{Float64})
    mx, my = landmark_pos[1], landmark_pos[2]
    mux, muy, muθ = pose[1], pose[2], pose[3]
    q = (mux - mx)^2 + (muy - my)^2
    q11 = (mux - mx) / sqrt(q)
    q12 = (muy - my) / sqrt(q)
    q21 = (my - muy) / q
    q22 = (mux - mx) / q
    return [q11 q12 0.0;
            q21 q22 -1.0]
end

function matQ(distance_dev::Float64, direction_dev::Float64)
    return Diagonal([distance_dev^2, direction_dev^2])
end

function motion_update(kf::KalmanFilter, v::Float64, ω::Float64, dt::Float64)
    if abs(ω) < 1e-5
        ω = 1e-5
    end
    μ = mean(kf.belief_)
    Σ = cov(kf.belief_)
    M = matM(v, ω, dt, kf.motion_noise_stds)
    A = matA(v, ω, dt, μ[3])
    F = matF(v, ω, dt, μ[3])
    cov_mat = F * Σ * transpose(F) + A * M * transpose(A)
    cov_mat = (cov_mat + transpose(cov_mat)) / 2.0
    kf.belief_ = MvNormal(state_transition(μ, v, ω, dt),
                          cov_mat)
    kf.pose_ = mean(kf.belief_)
end

function observation_update(kf::KalmanFilter, observation::Vector{Vector{Float64}})
    μ = mean(kf.belief_)
    Σ = cov(kf.belief_)
    envmap = kf.map_
    for obsv = observation
        z = obsv[1:2]
        obs_index = convert(Int64, obsv[3]) + 1 # landmark id starts from 0!

        H = matH(μ, envmap[obs_index].pos)
        estimated_z = observation_function(μ, envmap[obs_index].pos)
        Q = matQ(estimated_z[1] * kf.distance_dev_rate,
                 kf.direction_dev)
        K = Σ * transpose(H) * inv(Q + H * Σ * transpose(H))
        μ += K * (z - estimated_z)
        Σ = (Matrix(1.0I, 3, 3) - K * H) * Σ
    end
    Σ = (Σ + transpose(Σ)) / 2.0
    kf.belief_ = MvNormal(μ, Σ)
    kf.pose_ = μ
end

function draw(kf::KalmanFilter, p::Plot{T}) where T
    pose = kf.pose_
    cov_mat = cov(kf.belief_)
    p = covellipse!(pose[1:2], cov_mat[1:2, 1:2], n_std=3, aspect_ratio=1)

    x, y, θ = pose[1], pose[2], pose[3]
    sigma3 = sqrt(cov_mat[3, 3]) * 3.0
    xs = [x + cos(θ - sigma3), x, x + cos(θ + sigma3)]
    ys = [y + sin(θ - sigma3), y, y + sin(θ + sigma3)]
    p = plot!(xs, ys, color="blue", alpha=0.5)
    annota = "($(round(pose[1], sigdigits=3)), $(round(pose[2], sigdigits=3)), $(round(pose[3], sigdigits=3)))"
    p = annotate!(pose[1]+1.0, pose[2]+1.0, text(annota, 10))
end
