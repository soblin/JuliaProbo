@testset "ch05_mcl05" begin
    motion_noise_stds = Dict("vv" => 0.01, "vω" => 0.02, "ωv" => 0.03, "ωω" => 0.04)
    dt = 0.1
    
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    landmarks = [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    m = Map()
    push!(m, landmarks)

    # robot side
    initial_pose = [2.0, 2.0, pi/6]
    estimator = Mcl(initial_pose, 100, motion_noise_stds)
    circling = EstimatorAgent(0.2, 10.0 / 180 * pi, dt, estimator)
    robot = RealRobot(initial_pose, circling, RealCamera(landmarks); color="red")
    push!(world, robot)
    push!(world, m)
    anim = @animate for i in 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(nothing, robot.pose_)
        v, ω = decision(circling, z)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl05.gif", fps=10)
end

@testset "ch05_mcl07" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling = EstimatorAgent(0.2, 10.0*pi/180, dt, estimator)
    robot = RealRobot(initial_pose, circling, nothing; color="red")
    push!(world, robot)
    anim = @animate for i in 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_)
        v, ω = decision(circling, z)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl07.gif", fps=10)
end

@testset "ch05_mcl09" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    landmarks = [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    m = Map()
    push!(m, landmarks)
    world = World(xlim, ylim)
    push!(world, m)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0*pi/180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color="red")
    push!(world, robot)
    anim = @animate for i in 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise=true, bias=true)
        v, ω = decision(circling_agent, z, m)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl09.gif", fps=10)
end

@testset "ch05_mcl11" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0*pi/180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color="red")
    push!(world, robot)
    anim = @animate for i in 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise=true, bias=true)
        v, ω = decision(circling_agent, z, envmap)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl11.gif", fps=10)
end

@testset "ch05_mcl12" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0*pi/180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color="red")
    push!(world, robot)
    anim = @animate for i in 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise=true, bias=true)
        v, ω = decision(circling_agent, z, envmap; resample=true)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl12.gif", fps=10)
end

@testset "ch05_mcl13" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0*pi/180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color="red")
    push!(world, robot)
    anim = @animate for i in 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise=true, bias=true)
        v, ω = decision(circling_agent, z, envmap; resample=true)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl13.gif", fps=10)
end

@testset "ch05_mcl14" begin
    dt = 0.1
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1)]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, 0.0, 0.0]
    estimator = Mcl(initial_pose, 100)
    circling_agent = EstimatorAgent(0.2, 10.0*pi/180, dt, estimator)
    robot = RealRobot(initial_pose, circling_agent, RealCamera(landmarks); color="red")
    push!(world, robot)
    anim = @animate for i in 1:300
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(robot.sensor_, robot.pose_; noise=true, bias=true)
        v, ω = decision(circling_agent, z, envmap; resample=true)
        state_transition(robot, v, ω, dt; move_noise=true, vel_bias_noise=true)
    end
    #gif(anim, "ch05_mcl14.gif", fps=10)
end
