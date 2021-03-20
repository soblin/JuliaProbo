@testset "ch06_kf3" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator1 = KalmanFilter(envmap, initial_pose)
    agent1 = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator1)
    robot1 = RealRobot(initial_pose, agent1, RealCamera(landmarks); color = "red")
    push!(world, robot1)

    estimator2 = KalmanFilter(envmap, initial_pose)
    agent2 = EstimatorAgent(0.1, 0.0, dt, estimator2)
    robot2 = RealRobot(initial_pose, agent2, RealCamera(landmarks); color = "red")
    push!(world, robot2)

    estimator3 = KalmanFilter(envmap, initial_pose)
    agent3 = EstimatorAgent(0.1, -3.0 / 180 * pi, dt, estimator3)
    robot3 = RealRobot(initial_pose, agent3, RealCamera(landmarks); color = "red")
    push!(world, robot3)

    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        # update robot observation
        z1 = observations(robot1.sensor_, robot1.pose_; noise = true, bias = true)
        z2 = observations(robot2.sensor_, robot2.pose_; noise = true, bias = true)
        z3 = observations(robot3.sensor_, robot3.pose_; noise = true, bias = true)
        p = draw(world, annota)
        # robot1
        v1, ω1 = decision(agent1, z1, envmap)
        state_transition(robot1, v1, ω1, dt; move_noise = true, vel_bias_noise = true)
        # robot2
        v2, ω2 = decision(agent2, z2, envmap)
        state_transition(robot2, v2, ω2, dt; move_noise = true, vel_bias_noise = true)
        # robot3
        v3, ω3 = decision(agent3, z3, envmap)
        state_transition(robot3, v3, ω3, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch06_kf3.gif", fps = 20)
    end
end

@testset "ch06_kf4" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    n_robots = 3
    estimators = [
        KalmanFilter(envmap, initial_pose),
        KalmanFilter(envmap, initial_pose),
        KalmanFilter(envmap, initial_pose),
    ]
    agents = [
        EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimators[1]),
        EstimatorAgent(0.1, 0.0, dt, estimators[2]),
        EstimatorAgent(0.1, -3.0 / 180 * pi, dt, estimators[3]),
    ]
    robots = [
        RealRobot(initial_pose, agents[1], RealCamera(landmarks); color = "red"),
        RealRobot(initial_pose, agents[2], RealCamera(landmarks); color = "red"),
        RealRobot(initial_pose, agents[3], RealCamera(landmarks); color = "red"),
    ]
    for i = 1:n_robots
        push!(world, robots[i])
    end

    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        zs = []
        for i = 1:n_robots
            z = observations(robots[i].sensor_, robots[i].pose_; noise = true, bias = true)
            push!(zs, copy(z))
        end
        p = draw(world, annota)
        for i = 1:n_robots
            v, ω = decision(agents[i], zs[i], envmap)
            state_transition(robots[i], v, ω, dt; move_noise = true, vel_bias_noise = true)
        end
    end
    if GUI
        gif(anim, "ch06_kf4.gif", fps = 20)
    end
end
