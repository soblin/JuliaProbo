using Plots
using JuliaProbo

mutable struct IdealCamera <: AbstractSensor
    landmarks_::Vector{Landmark}
    last_observation_::Vector{Vector{Float64}}
    distance_range::Vector{Float64}
    direction_range::Vector{Float64}
    function IdealCamera(landmark::Vector{Landmark}, distance_range=[0.5, 6.0], direction_range=[-pi/3, pi/3])
        new(landmark, Vector{Vector{Float64}}[], distance_range, direction_range)
    end
end

function visible(camera::IdealCamera, polarpos::Vector{Float64}=nothing)
    if polarpos == nothing
        return false
    else
        return camera.distance_range[1] <= polarpos[1] <= camera.distance_range[2] &&
        camera.direction_range[1] <= polarpos[2] <= camera.direction_range[2]
    end
end

function observation_function(camera_pose::Vector{Float64}, landmark_pos::Vector{Float64})
    diff = landmark_pos .- camera_pose[1:2]
    ϕ = atan(diff[2], diff[1]) - camera_pose[3]
    while ϕ >= pi; ϕ -= 2 * pi end
    while ϕ < -pi; ϕ += 2 * pi end
    return [sqrt(sum(diff .* diff)), ϕ]
end

function observations(camera::IdealCamera, camera_pose::Vector{Float64})
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

function draw(camera::IdealCamera, camera_pose::Vector{Float64}, p)
    for obsv = camera.last_observation_
        x, y, θ = camera_pose
        distance, direction = obsv[1], obsv[2]
        lx = x + distance * cos(direction + θ)
        ly = y + distance * sin(direction + θ)
        p = plot!([x, lx], [y, ly], color="pink", legend=nothing)
    end
end
