using JuliaProbo
using Plots

function main()
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    world = World(xlim, ylim)
    circlings = Array{Agent, 1}(undef, 0)
    robots = Array{RealRobot, 1}(undef, 0)
    for i in 1:10
        circling = Agent(0.2, 10.0 / 180 * pi)
        robot = RealRobot([0.0, 0.0, 0.0], circling, nothing, 0.05, "black")
        push!(circlings, circling)
        push!(robots, robot)
        push!(world, robot)
    end

    dt = 0.1
    anim = @animate for i in 1:10
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        p = draw(world, annota);
        plot(p);
        for j in 1:10
            v, ω = decision(circlings[j], nothing)
            state_transition(robots[j], v, ω, dt)
        end
    end
    gif(anim, "ch4_sim2.gif", fps=10);
end

main()
