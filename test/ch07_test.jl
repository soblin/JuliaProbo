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
