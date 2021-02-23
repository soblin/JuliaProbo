using Plots

include("agent.jl")
include("sensor.jl")

mutable struct IdealRobot <: Object
    pose_::Vector{Float64}
    agent_::Agent
    radius_::Float64
    color_::String
    poses_::Vector{Vector{Float64}}
    sensor_::Sensor
    function IdealRobot(pose::Vector{Float64}, agent::Agent, sensor::Sensor, radius=0.2, color="blue")
        new([pose[1], pose[2], pose[3]], agent, radius, color, [copy(pose)], sensor)
    end
end

function draw(robot::IdealRobot, p)
    # robot
    p = scatter!([robot.pose_[1]], [robot.pose_[2]], markersize=robot.radius_ * 100, color=robot.color_, markeralpha=0.5, legend=nothing, aspect_ratio=:equal)
    θ = robot.pose_[3]
    # pose
    p = quiver!([robot.pose_[1]], [robot.pose_[2]], quiver=([robot.radius_ * cos(θ) * 5], [robot.radius_ * sin(θ) * 5]), color="black")
    # traj
    x_his = [pose[1] for pose in robot.poses_]
    y_his = [pose[2] for pose in robot.poses_]
    p = plot!(x_his, y_his, color="black", lw=0.5)
    # camera
    draw(robot.sensor_, robot.pose_, p)
end

function state_transition(robot::IdealRobot, v::Float64, ω::Float64, dt::Float64)
    θ = robot.pose_[3]
    new_pose = [0.0, 0.0, 0.0]
    if abs(ω) < 1e-10
        dpose = [v * cos(θ), v * sin(θ), ω] * dt
        new_pose = robot.pose_ .+ dpose
    else
        dpose = [v / ω * ( sin(θ + ω * dt) - sin(θ)), v / ω * ( -cos(θ + ω * dt) + cos(θ)), ω * dt]
        new_pose = robot.pose_ .+ dpose
    end
    push!(robot.poses_, copy(robot.pose_))
    robot.pose_ = copy(new_pose)
    return new_pose
end
