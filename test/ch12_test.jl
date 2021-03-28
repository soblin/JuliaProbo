@testset "ch12_dp_kf" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([1.0, 4.0], 0), Landmark([4.0, 1.0], 1), Landmark([-4.0, -4.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = PuddleWorld(xlim, ylim)
    push!(world, envmap)
    # goal
    goal = Goal(-3.0, -3.0)
    push!(world, goal)
    # robot side
    initial_pose = [2.0, 2.0, 0.0]
    estimator = KalmanFilter(envmap, initial_pose)
    reso = [0.1, 0.1, pi / 20]
    dp_agent = DpPolicyAgent(0.2, 10.0 * pi / 180, dt, estimator, goal, reso)
    init_policy(dp_agent, "ch12_policy.txt")
    robot = RealRobot(initial_pose, dp_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    # puddles
    push!(world, Puddle([-2.0, 0.0], [0.0, 2.0], 0.1))
    push!(world, Puddle([-0.5, -2.0], [2.5, 1.0], 0.1))

    anim = @animate for i = 1:100
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        # t
        update_status(world)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)

        # t+1
        v, ω = decision(dp_agent, z)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch12_dp_kf.gif", fps = 10)
    end
end

@testset "ch12_dp_mcl" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([1.0, 4.0], 0), Landmark([4.0, 1.0], 1), Landmark([-4.0, -4.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = PuddleWorld(xlim, ylim)
    push!(world, envmap)
    # goal
    goal = Goal(-3.0, -3.0)
    push!(world, goal)
    # robot side
    initial_pose = [2.0, 2.0, 0.0]
    # estimator = KalmanFilter(envmap, initial_pose)
    estimator = Mcl(initial_pose, 100)
    reso = [0.1, 0.1, pi / 20]
    dp_agent = DpPolicyAgent(0.2, 10.0 * pi / 180, dt, estimator, goal, reso)
    init_policy(dp_agent, "ch12_policy.txt")
    robot = RealRobot(initial_pose, dp_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    # puddles
    push!(world, Puddle([-2.0, 0.0], [0.0, 2.0], 0.1))
    push!(world, Puddle([-0.5, -2.0], [2.5, 1.0], 0.1))

    anim = @animate for i = 1:180
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        # t
        update_status(world)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)

        # t+1
        v, ω = decision(dp_agent, z, envmap; resample = true)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch12_dp_mcl.gif", fps = 10)
    end
end

@testset "ch12_qmdp12" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks =
        [Landmark([1.0, 4.0], 0), Landmark([4.0, 1.0], 1), Landmark([-4.0, -4.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = PuddleWorld(xlim, ylim)
    push!(world, envmap)
    # goal
    goal = Goal(-3.0, -3.0)
    push!(world, goal)
    # robot side
    initial_pose = [2.0, 2.0, 0.0]
    # estimator = KalmanFilter(envmap, initial_pose)
    estimator = Mcl(initial_pose, 100)
    reso = [0.1, 0.1, pi / 20]
    dp_agent = QMDPAgent(0.2, 10.0 * pi / 180, dt, estimator, goal, reso)
    init_policy(dp_agent, "ch12_policy.txt")
    init_value(dp_agent, "ch12_value.txt")
    robot = RealRobot(initial_pose, dp_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    # puddles
    push!(world, Puddle([-2.0, 0.0], [0.0, 2.0], 0.1))
    push!(world, Puddle([-0.5, -2.0], [2.5, 1.0], 0.1))

    anim = @animate for i = 1:180
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        # t
        update_status(world)
        z = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        p = draw(world, annota)

        # t+1
        v, ω = decision(dp_agent, z, envmap; resample = true)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch12_qmdp1.gif", fps = 10)
    end
end
