using JuliaProbo

mutable struct IdealRobot <: AbstractObject
    pose_::Vector{Float64}
    agent_::AbstractAgent
    radius_::Float64
    color_::String
    poses_::Vector{Vector{Float64}}
    sensor_::Union{AbstractSensor,Nothing}
    function IdealRobot(
        pose::Vector{Float64},
        agent::AbstractAgent,
        sensor::Union{AbstractSensor,Nothing},
        radius = 0.05,
        color = "blue",
    )
        if typeof(sensor) == Nothing
            new([pose[1], pose[2], pose[3]], agent, radius, color, [copy(pose)], nothing)
        else
            new([pose[1], pose[2], pose[3]], agent, radius, color, [copy(pose)], sensor)
        end
    end
end

function draw(robot::IdealRobot, p::Plot{T}) where {T}
    # robot
    p = scatter!(
        [robot.pose_[1]],
        [robot.pose_[2]],
        markersize = robot.radius_ * 100,
        color = robot.color_,
        markeralpha = 0.5,
        legend = nothing,
        aspect_ratio = :equal,
    )
    θ = robot.pose_[3]
    # pose
    p = quiver!(
        [robot.pose_[1]],
        [robot.pose_[2]],
        quiver = ([robot.radius_ * cos(θ) * 5], [robot.radius_ * sin(θ) * 5]),
        color = "black",
    )
    # traj
    x_his = [pose[1] for pose in robot.poses_]
    y_his = [pose[2] for pose in robot.poses_]
    p = plot!(x_his, y_his, color = "black", lw = 0.5)
    # camera
    draw(robot.sensor_, robot.pose_, p)
end

function state_transition(cur_pose::Vector{Float64}, v::Float64, ω::Float64, dt::Float64)
    θ = cur_pose[3]
    new_pose = [0.0, 0.0, 0.0]
    if abs(ω) < 1e-10
        dpose = [v * cos(θ), v * sin(θ), ω] * dt
        new_pose = cur_pose .+ dpose
    else
        dpose = [
            v / ω * (sin(θ + ω * dt) - sin(θ)),
            v / ω * (-cos(θ + ω * dt) + cos(θ)),
            ω * dt,
        ]
        new_pose = cur_pose .+ dpose
    end
    return new_pose
end

function state_transition(robot::IdealRobot, v::Float64, ω::Float64, dt::Float64)
    θ = robot.pose_[3]
    new_pose = [0.0, 0.0, 0.0]
    if abs(ω) < 1e-10
        dpose = [v * cos(θ), v * sin(θ), ω] * dt
        new_pose = robot.pose_ .+ dpose
    else
        dpose = [
            v / ω * (sin(θ + ω * dt) - sin(θ)),
            v / ω * (-cos(θ + ω * dt) + cos(θ)),
            ω * dt,
        ]
        new_pose = robot.pose_ .+ dpose
    end
    push!(robot.poses_, copy(robot.pose_))
    robot.pose_ = copy(new_pose)
    return new_pose
end

struct PoseUniform
    low::Vector{Float64}
    upp::Vector{Float64}
    uni::Uniform{Float64}
    function PoseUniform(xlim::Vector{Float64}, ylim::Vector{Float64})
        new([xlim[1], ylim[1], 0], [xlim[2], ylim[2], 2 * pi], Uniform())
    end
end

function uniform(mv::PoseUniform)
    c = mv.upp - mv.low
    x = [rand(mv.uni) for i = 1:3]
    return mv.low .+ (x .* c)
end

mutable struct RealRobot <: AbstractObject
    pose_::Vector{Float64}
    agent_::AbstractAgent
    radius_::Float64
    color_::String
    poses_::Vector{Vector{Float64}}
    sensor_::Union{AbstractSensor,Nothing}
    # motion uncertainty
    ## movement noise
    noise_::Exponential{Float64}
    theta_noise_::Normal{Float64}
    distance_until_noise_::Float64
    ## velocity bias noise
    bias_rate_v_::Float64
    bias_rate_ω_::Float64
    ## stuck noise
    stuck_noise_::Exponential{Float64}
    escape_noise_::Exponential{Float64}
    time_until_stuck_::Float64
    time_until_escape_::Float64
    is_stuck_::Bool
    ## kidnap
    kidnap_noise_::Exponential{Float64}
    time_until_kidnap_::Float64
    kidnap_distrib_::PoseUniform

    function RealRobot(
        pose::Vector{Float64},
        agent::AbstractAgent,
        sensor::Union{AbstractSensor,Nothing};
        radius = 0.05,
        color = "blue",
        noise_per_meter = 5.0,
        noise_std = pi / 60,
        bias_rate_stds = (0.1, 0.1),
        expected_stuck_time = 1e100,
        expected_escape_time = 1e-100,
        expected_kidnap_time = 1e100,
        kidnap_range_x = [-5.0, 5.0],
        kidnap_range_y = [-5.0, 5.0],
    )
        noise = Exponential(1.0 / (1e-100 + noise_per_meter))
        stuck_noise = Exponential(expected_stuck_time)
        escape_noise = Exponential(expected_escape_time)
        kidnap_noise = Exponential(expected_kidnap_time)

        new(
            [pose[1], pose[2], pose[3]],
            agent,
            radius,
            color,
            [copy(pose)],
            sensor,
            noise, ## movement noise
            Normal(0.0, noise_std),
            rand(noise),
            rand(Normal(1.0, bias_rate_stds[1])), ## velocity noise
            rand(Normal(1.0, bias_rate_stds[2])),
            stuck_noise, ## stuck noise
            escape_noise,
            rand(stuck_noise),
            rand(escape_noise),
            false,
            kidnap_noise, ## kidnap
            rand(kidnap_noise),
            PoseUniform(kidnap_range_x, kidnap_range_y),
        )
    end
end

function apply_bias_error(robot::RealRobot, v::Float64, ω::Float64)
    return v * robot.bias_rate_v_, ω * robot.bias_rate_ω_
end

function apply_stuck_error(robot::RealRobot, v::Float64, ω::Float64, dt::Float64)
    if robot.is_stuck_
        robot.time_until_escape_ -= dt
        if robot.time_until_escape_ <= 0.0
            robot.time_until_escape_ += rand(robot.escape_noise_)
            robot.is_stuck_ = false
        end
    else
        robot.time_until_stuck_ -= dt
        if robot.time_until_stuck_ <= 0.0
            robot.time_until_stuck_ += rand(robot.stuck_noise_)
            robot.is_stuck_ = true
        end
    end

    return v * convert(Float64, robot.is_stuck_), ω * convert(Float64, robot.is_stuck_)
end

function apply_kidnap(robot::RealRobot, dt::Float64)
    robot.time_until_kidnap_ -= dt
    if robot.time_until_kidnap_ <= 0.0
        robot.time_until_kidnap_ += rand(robot.kidnap_noise_)
        robot.pose_ = uniform(robot.kidnap_distrib_)
    end
end

function state_transition(
    robot::RealRobot,
    v_::Float64,
    ω_::Float64,
    dt::Float64;
    move_noise = false,
    vel_bias_noise = false,
    stuck_noise = false,
    kidnap = false,
)
    v = v_
    ω = ω_

    ## velocity bias  noise
    if vel_bias_noise
        v, ω = apply_bias_error(robot, v, ω)
    end

    ## stuck noise
    if stuck_noise
        v, ω = apply_stuck_error(robot, v, ω, dt)
    end

    θ = robot.pose_[3]
    new_pose = [0.0, 0.0, 0.0]
    if abs(ω) < 1e-10
        dpose = [v * cos(θ), v * sin(θ), ω] * dt
        new_pose = robot.pose_ .+ dpose
    else
        dpose = [
            v / ω * (sin(θ + ω * dt) - sin(θ)),
            v / ω * (-cos(θ + ω * dt) + cos(θ)),
            ω * dt,
        ]
        new_pose = robot.pose_ .+ dpose
    end

    ## movement noise
    if move_noise
        robot.distance_until_noise_ -= (abs(v) * dt + robot.radius_ * abs(ω) * dt)
        if robot.distance_until_noise_ <= 0.0
            robot.distance_until_noise_ += rand(robot.noise_)
            new_pose[3] += rand(robot.theta_noise_)
        end
    end

    # save trajectory
    push!(robot.poses_, copy(robot.pose_))

    robot.pose_ = copy(new_pose)

    ## kidnap
    if kidnap
        apply_kidnap(robot, dt)
    end
end

function draw(robot::RealRobot, p::Plot{T}) where {T}
    # robot
    p = scatter!(
        [robot.pose_[1]],
        [robot.pose_[2]],
        markersize = robot.radius_ * 100,
        color = robot.color_,
        markeralpha = 0.5,
        legend = nothing,
        aspect_ratio = :equal,
    )
    θ = robot.pose_[3]
    # pose
    p = quiver!(
        [robot.pose_[1]],
        [robot.pose_[2]],
        quiver = ([robot.radius_ * cos(θ) * 5], [robot.radius_ * sin(θ) * 5]),
        color = "black",
    )
    # traj
    x_his = [pose[1] for pose in robot.poses_]
    y_his = [pose[2] for pose in robot.poses_]
    p = plot!(x_his, y_his, color = "black", lw = 0.5)
    # camera
    draw(robot.sensor_, robot.pose_, p)
    # agent
    draw(robot.agent_, p)
end
