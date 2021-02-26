using JuliaProbo

mutable struct IdealCamera <: AbstractSensor
    landmarks_::Vector{Landmark}
    last_observation_::Vector{Vector{Float64}}
    distance_range::Vector{Float64}
    direction_range::Vector{Float64}
    function IdealCamera(landmark::Vector{Landmark},
                         distance_range=[0.5, 6.0],
                         direction_range=[-pi/3, pi/3])
        new(landmark,
            Vector{Vector{Float64}}[],
            distance_range,
            direction_range)
    end
end

function visible(camera::IdealCamera, polarpos::Vector{Float64})
    return camera.distance_range[1] <= polarpos[1] <= camera.distance_range[2] &&
        camera.direction_range[1] <= polarpos[2] <= camera.direction_range[2]
end

function observation_function(camera_pose::Vector{Float64}, landmark_pos::Vector{Float64})
    diff = landmark_pos .- camera_pose[1:2]
    ϕ = atan(diff[2], diff[1]) - camera_pose[3]
    while ϕ >= pi; ϕ -= 2 * pi end
    while ϕ < -pi; ϕ += 2 * pi end
    return [sqrt(sum(diff .* diff)), ϕ]
end

function observations(camera::Union{IdealCamera, Nothing}, camera_pose::Vector{Float64})
    if typeof(camera) == Nothing
        return nothing
    end
    n = size(camera.landmarks_)[1]
    observed = [[1.0]]
    pop!(observed)
    for i in 1:n
        z = observation_function(camera_pose, camera.landmarks_[i].pos)
        if visible(camera, z)
            push!(observed, copy(z))
        end
    end
    camera.last_observation_ = copy(observed)
    return observed
end

function draw(camera::Union{IdealCamera, Nothing}, camera_pose::Vector{Float64}, p::Plot{T}) where T
    if camera == nothing
        return
    end
    for obsv = camera.last_observation_
        x, y, θ = camera_pose
        distance, direction = obsv[1], obsv[2]
        lx = x + distance * cos(direction + θ)
        ly = y + distance * sin(direction + θ)
        p = plot!([x, lx], [y, ly], color="pink", legend=nothing)
    end
end

mutable struct RealCamera <: AbstractSensor
    landmarks_::Vector{Landmark}
    last_observation_::Vector{Vector{Float64}}
    distance_range::Vector{Float64}
    direction_range::Vector{Float64}
    distance_noise_rate::Float64
    direction_noise::Float64
    distance_bias_rate_std::Float64
    direction_bias::Float64
    function RealCamera(landmarks::Vector{Landmark},
                        distance_range=[0.5, 6.0],
                        direction_range=[-pi/3, pi/3];
                        distance_noise_rate=0.1,
                        direction_noise=pi/90,
                        distance_bias_rate_stddev=0.1,
                        direction_bias_stddev=pi/90)

        new(landmarks,
            Vector{Vector{Float64}}(undef, 0),
            distance_range,
            direction_range,
            distance_noise_rate,
            direction_noise,
            rand(Normal(0.0, distance_bias_rate_stddev)),
            rand(Normal(0.0, direction_bias_stddev))
            )
    end
end

function visible(camera::RealCamera, polarpos::Vector{Float64})
    return camera.distance_range[1] <= polarpos[1] <= camera.distance_range[2] &&
        camera.direction_range[1] <= polarpos[2] <= camera.direction_range[2]
end

function apply_noise(camera::RealCamera, z::Vector{Float64})
    d = z[1]
    ϕ = z[2]
    errored_d = rand(Normal(d, d * camera.distance_noise_rate))
    errored_ϕ = rand(Normal(ϕ, camera.direction_noise))
    return [errored_d, errored_ϕ]
end

function apply_bias(camera::RealCamera, z::Vector{Float64})
    d = z[1]
    ϕ = z[2]
    d += d * camera.distance_bias_rate_std
    ϕ += camera.direction_bias
    return [d, ϕ]
end

function observations(camera::RealCamera, camera_pose::Vector{Float64})
    n = size(camera.landmarks_)[1]
    observed = [[1.0]]
    pop!(observed)
    for i in 1:n
        z = observation_function(camera_pose, camera.landmarks_[i].pos)
        if visible(camera, z)
            z = apply_bias(camera, z)
            z = apply_noise(camera, z)
            push!(observed, [z[1], z[2], camera.landmarks_[i].id])
        end
    end
    camera.last_observation_ = copy(observed)
    return observed
end

function draw(camera::RealCamera, camera_pose::Vector{Float64}, p::Plot{T}) where T
    for obsv = camera.last_observation_
        x, y, θ = camera_pose
        distance, direction = obsv[1], obsv[2]
        lx = x + distance * cos(direction + θ)
        ly = y + distance * sin(direction + θ)
        p = plot!([x, lx], [y, ly], color="pink", legend=nothing)
    end
end
