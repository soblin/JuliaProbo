@testset "ch07_kdl_mcl" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [Landmark([2.0, -3.0], 0), Landmark([3.0, 3.0], 1)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = KdlMcl(initial_pose, 1000)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    anim = @animate for i = 1:10
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        v, ω = decision(circling_agent, z, envmap)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    #gif(anim, "ch07_kdl_mcl.gif", fps=10)
end

@testset "ch07_kdl_mcl" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([2.0, -3.0], 0), Landmark([3.0, 3.0], 1), Landmark([-4.0, 2.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100; glob = true, xlim = xlim, ylim = ylim)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    anim = @animate for i = 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        v, ω = decision(circling_agent, z, envmap; resample = true)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    # gif(anim, "images/ch07_mcl_global.gif", fps=10)
end

@testset "ch07_kf_global" begin
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
    estimator = KalmanFilter(envmap, initial_pose; glob = true, xlim = xlim, ylim = ylim)
    agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, agent, RealCamera(landmarks); color = "red")
    push!(world, robot)

    anim = @animate for i = 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        v, ω = decision(agent, z, envmap)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    #gif(anim, "images/ch07_kf_global.gif", fps=10)
end

@testset "ch07_kf_kidnap" begin
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
    estimator = KalmanFilter(envmap, initial_pose)
    agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(
        initial_pose,
        agent,
        RealCamera(landmarks);
        color = "red",
        expected_kidnap_time = 10,
    )
    push!(world, robot)

    anim = @animate for i = 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        v, ω = decision(agent, z, envmap)
        state_transition(
            robot,
            v,
            ω,
            dt;
            move_noise = true,
            vel_bias_noise = true,
            kidnap = true,
        )
    end
    #gif(anim, "images/ch07_kf_global.gif", fps=10)
end

@testset "ch07_mcl_kidnap" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([2.0, -3.0], 0), Landmark([3.0, 3.0], 1), Landmark([-4.0, 2.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(
        initial_pose,
        circling_agent,
        RealCamera(landmarks);
        color = "red",
        expected_kidnap_time = 20,
    )
    push!(world, robot)
    anim = @animate for i = 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        v, ω = decision(circling_agent, z, envmap; resample = true)
        state_transition(
            robot,
            v,
            ω,
            dt;
            move_noise = true,
            vel_bias_noise = true,
            kidnap = true,
        )
    end
    #gif(anim, "images/ch07_mcl_kidnap.gif", fps=10)
end
