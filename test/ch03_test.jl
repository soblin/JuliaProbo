using Plots

@testset "ch03_robot11" begin
    straight_agent = Agent(0.2, 0.0)
    circling_agent = Agent(0.2, 10.0 / 180.0 * pi)
    landmarks = [Landmark([2.0, -2.0], 0), Landmark([-1.0, -3.0], 1), Landmark([3.0, 3.0], 2)]
    m = Map()
    camera1 = IdealCamera(landmarks)
    camera2 = IdealCamera(landmarks)
    push!(m, landmarks)
    robot1 = IdealRobot([2.0, 3.0, pi/6], straight_agent, camera1, 0.05, "blue",)
    robot2 = IdealRobot([-2.0, -1.0, pi/5*6], circling_agent, camera2, 0.05, "orange")
    xlim = [-5.5, 10]
    ylim = [-5.5, 10]
    world = World(xlim, ylim)
    push!(world, robot1)
    push!(world, robot2)
    push!(world, m)
    dt = 0.1
    anim = @animate for i = 1:50
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        obsv1 = observations(camera1, robot1.pose_)
        obsv2 = observations(camera2, robot2.pose_)
        p = draw(world, annota)
        plot(p)
        v, ω = decision(straight_agent, obsv1)
        state_transition(robot1, v, ω, dt)
        v, ω = decision(circling_agent, obsv2)
        state_transition(robot2, v, ω, dt)
    end
    gif(anim, "ch3_robot11.gif", fps=10);
end
