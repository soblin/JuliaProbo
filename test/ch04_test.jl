@testset "ch04_sim02" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    circlings = Array{Agent,1}(undef, 0)
    robots = Array{RealRobot,1}(undef, 0)
    for i = 1:10
        circling = Agent(0.2, 10.0 / 180 * pi)
        robot =
            RealRobot([0.0, 0.0, 0.0], circling, nothing; radius = 0.05, color = "black")
        push!(circlings, circling)
        push!(robots, robot)
        push!(world, robot)
    end

    dt = 0.1
    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        plot(p)
        for j = 1:10
            obsv = observations(robots[j].sensor_, robots[j].pose_)
            @assert obsv == nothing
            v, ω = decision(circlings[j], obsv)
            state_transition(robots[j], v, ω, dt; move_noise = true)
        end
    end
    #gif(anim, "ch04_sim02.gif", fps=10);
end

@testset "ch04_sim03" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    circling_agent = Agent(0.2, 10.0 / 180 * pi)
    nobias_robot = IdealRobot([0.0, 0.0, 0.0], circling_agent, nothing, 0.05, "gray")
    push!(world, nobias_robot)
    biased_robot = RealRobot(
        [0.0, 0.0, 0.0],
        circling_agent,
        nothing;
        radius = 0.05,
        color = "red",
        bias_rate_stds = (0.2, 0.2),
    )
    push!(world, biased_robot)
    dt = 0.1
    anim = @animate for i = 1:50
        annota = "t = $(round(dt * i, sigdigits=3))[s]"
        p = draw(world, annota)
        obsv1 = observations(nobias_robot.sensor_, nobias_robot.pose_)
        v1, ω1 = decision(circling_agent, obsv1)
        obsv2 = observations(biased_robot.sensor_, biased_robot.pose_)
        v2, ω2 = decision(circling_agent, obsv2)
        state_transition(nobias_robot, v1, ω1, dt)
        state_transition(biased_robot, v2, ω2, dt; vel_bias_noise = true)
    end
    #gif(anim, "ch04_sim03.gif", fps=10)
end

@testset "ch04_sim04" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)

    circling_agent = Agent(0.2, 10.0 / 180 * pi)
    robots = Array{RealRobot,1}(undef, 0)
    for i = 1:10
        robot = RealRobot(
            [0.0, 0.0, 0.0],
            circling_agent,
            nothing;
            radius = 0.05,
            color = "gray",
            expected_stuck_time = 60.0,
            expected_escape_time = 60.0,
        )
        push!(robots, robot)
        push!(world, robot)
    end
    ideal_robot = IdealRobot([0.0, 0.0, 0.0], circling_agent, nothing, 0.05, "red")
    push!(world, ideal_robot)

    dt = 0.1
    anim = @animate for i = 1:50
        annota = "t = $(round(dt * i, sigdigits=3))[s]"
        p = draw(world, annota)
        for j = 1:10
            obsv = observations(robots[j].sensor_, robots[j].pose_)
            v, ω = decision(circling_agent, obsv)
            state_transition(robots[j], v, ω, dt; stuck_noise = true)
        end
        obsv = observations(ideal_robot.sensor_, ideal_robot.pose_)
        v, ω = decision(circling_agent, obsv)
        state_transition(ideal_robot, v, ω, dt)
    end
    #gif(anim, "ch04_sim04.gif", fps=10)
end

@testset "ch04_sim05" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)

    circling_agent = Agent(0.2, 10.0 / 180 * pi)
    robots = Array{RealRobot,1}(undef, 0)
    for i = 1:10
        robot = RealRobot(
            [0.0, 0.0, 0.0],
            circling_agent,
            nothing;
            radius = 0.05,
            color = "gray",
            expected_kidnap_time = 5.0,
        )
        push!(robots, robot)
        push!(world, robot)
    end
    ideal_robot = IdealRobot([0.0, 0.0, 0.0], circling_agent, nothing, 0.05, "red")
    push!(world, ideal_robot)

    dt = 0.1
    anim = @animate for i = 1:50
        annota = "t = $(round(dt * i, sigdigits=3))[s]"
        p = draw(world, annota)
        for j = 1:10
            obsv = observations(robots[j].sensor_, robots[j].pose_)
            v, ω = decision(circling_agent, obsv)
            state_transition(robots[j], v, ω, dt; kidnap = true)
        end
        obsv = observations(ideal_robot.sensor_, ideal_robot.pose_)
        v, ω = decision(circling_agent, obsv)
        state_transition(ideal_robot, v, ω, dt)
    end
    #gif(anim, "ch04_sim05.gif", fps=10)
end

@testset "ch04_sim07" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    circling_agent = Agent(0.2, 10.0 / 180 * pi)
    landmarks =
        [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 1)]
    m = Map()
    push!(m, landmarks)
    robot =
        RealRobot([0.0, 0.0, pi / 6], circling_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    push!(world, m)
    dt = 0.1
    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        plot(p)
        obsv = observations(robot.sensor_, robot.pose_; noise = true)
        v, ω = decision(circling_agent, obsv)
        state_transition(robot, v, ω, dt)
    end
    #gif(anim, "ch04_sim07.gif", fps=10)
end

@testset "ch04_sim08" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    straight_agent = Agent(0.2, 0.0)
    landmarks = [
        Landmark([-4.0, 2.0], 0),
        Landmark([3.0, -3.0], 1),
        Landmark([3.0, 3.0], 2),
        Landmark([3.0, -2.0], 3),
        Landmark([3.0, 0.0], 4),
        Landmark([3.0, 1.0], 5),
    ]
    m = Map()
    push!(m, landmarks)
    robot = RealRobot([0.0, 0.0, 0.0], straight_agent, RealCamera(landmarks); color = "red")
    push!(world, robot)
    push!(world, m)
    dt = 0.1
    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        plot(p)
        obsv = observations(robot.sensor_, robot.pose_; noise = true, bias = true)
        v, ω = decision(straight_agent, obsv)
        state_transition(robot, v, ω, dt)
    end
    #gif(anim, "ch04_sim08.gif", fps=10)
end

@testset "ch04_sim09" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    landmarks =
        [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    m = Map()
    push!(m, landmarks)
    circling = Agent(0.2, 10.0 / 180 * pi)
    robot = RealRobot(
        [0.0, 0.0, 0.0],
        circling,
        RealCamera(landmarks; phantom_prob = 0.2);
        color = "red",
    )
    push!(world, robot)
    push!(world, m)
    dt = 0.1
    anim = @animate for i = 1:100
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(
            robot.sensor_,
            robot.pose_;
            noise = true,
            bias = true,
            phantom = true,
        )
        v, ω = decision(circling, z)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
end

@testset "ch04_sim10" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    landmarks =
        [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    m = Map()
    push!(m, landmarks)
    circling = Agent(0.2, 10.0 / 180 * pi)
    robot = RealRobot(
        [0.0, 0.0, 0.0],
        circling,
        RealCamera(landmarks; overlook_prob = 0.5);
        color = "red",
    )
    push!(world, robot)
    push!(world, m)
    dt = 0.1
    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(
            robot.sensor_,
            robot.pose_;
            noise = true,
            bias = true,
            overlook = true,
        )
        v, ω = decision(circling, z)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
end

@testset "ch04_sim11" begin
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    landmarks =
        [Landmark([-4.0, 2.0], 0), Landmark([2.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    m = Map()
    push!(m, landmarks)
    circling = Agent(0.2, 10.0 / 180 * pi)
    robot = RealRobot(
        [2.0, 2.0, pi / 6],
        circling,
        RealCamera(landmarks; occlusion_prob = 0.1);
        color = "red",
    )
    push!(world, robot)
    push!(world, m)
    dt = 0.1
    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota)
        z = observations(
            robot.sensor_,
            robot.pose_;
            noise = true,
            bias = true,
            occlusion = true,
        )
        v, ω = decision(circling, z)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    #gif(anim, "ch04_sim11.gif", fps=10)
end
