@testset "ch07_kld_mcl" begin
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
    estimator = KldMcl(initial_pose, 1000)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
        v, ω = decision(circling_agent, z, envmap; resample = true)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch07_kld_mcl.gif", fps = 20)
    end
end

@testset "ch07_mcl_global" begin
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
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
        v, ω = decision(circling_agent, z, envmap; resample = true)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch07_mcl_global.gif", fps = 20)
    end
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

    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
        v, ω = decision(agent, z, envmap)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch07_kf_global.gif", fps = 20)
    end
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

    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
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
    if GUI
        gif(anim, "ch07_kf_global.gif", fps = 20)
    end
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
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
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
    if GUI
        gif(anim, "ch07_mcl_kidnap.gif", fps = 20)
    end
end

@testset "ch07_reset_mcl" begin
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
    estimator = ResetMcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color = "red")

    push!(world, robot)
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
        v, ω = decision(circling_agent, z, envmap; resample = true, sensor_reset=true)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch07_reset_mcl.gif", fps = 20)
    end
end

@testset "ch07_sensor_reset" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [
        Landmark([2.0, -3.0], 0),
        Landmark([3.0, 3.0], 1),
        Landmark([-4.0, 2.0], 2),
        Landmark([0.0, 0.0], 3),
        Landmark([-4.0, -4.0], 4),
    ]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = uniform(PoseUniform(xlim, ylim))
    estimator = ResetMcl(initial_pose, 100; xlim = xlim, ylim = ylim, α_threshold = 0.005)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(
        initial_pose,
        circling_agent,
        RealCamera(landmarks);
        color = "red",
        expected_kidnap_time = 10,
    )
    push!(world, robot)
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
        v, ω = decision(circling_agent, z, envmap; resample = true, sensor_reset = true)
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
    if GUI
        gif(anim, "ch07_sensor_reset2.gif", fps = 20)
    end
end

@testset "ch07_reset_mcl2" begin
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
    estimator = ResetMcl(initial_pose, 100; xlim = xlim, ylim = ylim, α_threshold = 0.005)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(
        initial_pose,
        circling_agent,
        RealCamera(landmarks);
        color = "red",
        expected_kidnap_time = 30,
    )
    push!(world, robot)
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)
        v, ω = decision(circling_agent, z, envmap; resample = true, sensor_reset = true)
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
    if GUI
        gif(anim, "ch07_reset_mcl2.gif", fps = 20)
    end
end

@testset "ch07_adaptive_mcl" begin
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
    estimator = AMcl(initial_pose, 100; xlim = xlim, ylim = ylim)
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(
        initial_pose,
        circling_agent,
        RealCamera(
            landmarks,
            phantom_prob = 0.1,
            phantom_range_x = xlim,
            phantom_range_y = ylim,
        );
        color = "red",
    )
    push!(world, robot)
    anim = @animate for i = 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(
            robot.sensor_,
            robot.pose_;
            noise = true,
            bias = true,
            phantom = true,
        )
        p = draw(world, annota)
        v, ω = decision(circling_agent, z, envmap)
        state_transition(
            robot,
            v,
            ω,
            dt;
            move_noise = true,
            vel_bias_noise = true,
            kidnap = false,
        )
    end
    if GUI
        gif(anim, "ch07_adaptive_mcl.gif", fps = 20)
    end
end
