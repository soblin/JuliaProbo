@testset "ch10_puddle_world2" begin
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
        RealCamera(landmarks, distance_bias_rate_stddev = 0.0, direction_bias_stddev = 0.0);
        color = "red",
    )
    push!(world, robot)
    # goal
    goal = Goal(-3.0, -3.0)
    push!(world, goal)
    # puddles
    push!(world, Puddle([-2.0, 0.0], [0.0, 2.0], 0.1))
    push!(world, Puddle([-0.5, -2.0], [2.5, 1.0], 0.1))

    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_; noise = false, bias = false)
        p = draw(world, annota)
        v, ω = decision(agent, z, envmap)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch10_puddle_world2.gif", fps = 20)
    end
end
