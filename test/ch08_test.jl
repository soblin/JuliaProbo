@testset "ch08_fastslam02345" begin
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
    estimator = FastSlam1(initial_pose, 100, length(landmarks))
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    anim = @animate for i = 1:400
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
        gif(anim, "ch08_fastslam02345.gif", fps = 20)
    end
end

@testset "ch08_fastslam067" begin
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
    estimator = FastSlam2(initial_pose, 100, length(landmarks))
    circling_agent = EstimatorAgent(0.2, 10.0 * pi / 180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    anim = @animate for i = 1:400
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
        gif(anim, "ch08_fastslam067.gif", fps = 20)
    end
end
