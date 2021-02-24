using JuliaProbo
using Plots

function ch3_robot11()
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
    anim = @animate for i = 1:5
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

function ch4_sim2()
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    circlings = Array{Agent, 1}(undef, 0)
    robots = Array{RealRobot, 1}(undef, 0)
    for i in 1:10
        circling = Agent(0.2, 10.0 / 180 * pi)
        robot = RealRobot([0.0, 0.0, 0.0], circling, nothing; radius=0.05, color="black", bias_rate_stds=(0.0, 0.0))
        push!(circlings, circling)
        push!(robots, robot)
        push!(world, robot)
    end

    dt = 0.1
    anim = @animate for i in 1:5
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota);
        plot(p);
        for j in 1:10
            obsv = observations(robots[j].sensor_, robots[j].pose_)
            @assert obsv == nothing
            v, ω = decision(circlings[j], obsv)
            state_transition(robots[j], v, ω, dt)
        end
    end
    gif(anim, "ch4_sim2.gif", fps=10);
end

function main()
    ch3_robot11()
    ch4_sim2()
end

main()
