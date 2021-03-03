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

struct Uniform2D
    low::Vector{Float64}
    upp::Vector{Float64}
    uni::Uniform{Float64}
    function Uniform2D(x::Vector{Float64}, y::Vector{Float64})
        new([x[1], y[1]], [x[2], y[2]], Uniform())
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
    phantom_prob::Float64
    phantom_distrib::Uniform2D
    overlook_prob::Float64
    occlusion_prob::Float64
    function RealCamera(landmarks::Vector{Landmark},
                        distance_range=[0.5, 6.0],
                        direction_range=[-pi/3, pi/3];
                        distance_noise_rate=0.1,
                        direction_noise=pi/90,
                        distance_bias_rate_stddev=0.1,
                        direction_bias_stddev=pi/90,
                        phantom_prob=0.0,
                        phantom_range_x=[-5.0, 5.0],
                        phantom_range_y=[-5.0, 5.0],
                        overlook_prob=0.1,
                        occlusion_prob=0.0)

        new(landmarks,
            Vector{Vector{Float64}}(undef, 0),
            distance_range,
            direction_range,
            distance_noise_rate,
            direction_noise,
            rand(Normal(0.0, distance_bias_rate_stddev)),
            rand(Normal(0.0, direction_bias_stddev)),
            phantom_prob,
            Uniform2D(phantom_range_x, phantom_range_y),
            overlook_prob,
            occlusion_prob
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

function gen_phantom(dist::Uniform2D)
    c = dist.upp - dist.low
    x = [rand(dist.uni) for i in 1:2]
    return dist.low .+ (x .* c)
end

function if_overlook(camera::RealCamera)
    if rand(Uniform()) < camera.overlook_prob
        return true
    else
        return false
    end
end

function apply_occlusion(camera::RealCamera, z::Vector{Float64})
    if rand(Uniform()) < camera.occlusion_prob
        return [z[1] + rand(Uniform()) * (camera.distance_range[2] - z[1]),
                z[2]]
    else
        return z
    end
end

function observations(camera::RealCamera, camera_pose::Vector{Float64}; noise=false, bias=false, phantom=false, overlook=false, occlusion=false)
    n = size(camera.landmarks_)[1]
    observed = [[1.0]]
    pop!(observed)
    for i in 1:n
        z = observation_function(camera_pose, camera.landmarks_[i].pos)
        if phantom
            if rand(Uniform()) < camera.phantom_prob
                phantom_pos = gen_phantom(camera.phantom_distrib)
                z = observation_function(camera_pose, phantom_pos)
            end
        end
        if overlook && if_overlook(camera)
            # not added
            continue
        end
        if occlusion
            z = apply_occlusion(camera, z)
        end
        if visible(camera, z)
            if bias
                z = apply_bias(camera, z)
            end
            if noise
                z = apply_noise(camera, z)
            end
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
