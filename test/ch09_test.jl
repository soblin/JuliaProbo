@testset "ch09_logger2" begin
    dt = 3.0
    # environment
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    # id of landmark must start from 0 with 1 step
    landmarks = [
        Landmark([-4.0, 2.0], 0),
        Landmark([2.0, -3.0], 1),
        Landmark([3.0, 3.0], 2),
        Landmark([0.0, 4.0], 3),
        Landmark([1.0, 1.0], 4),
        Landmark([-3.0, -1.0], 5),
    ]
    envmap = Map()
    push!(envmap, landmarks)
    world = World(xlim, ylim)
    push!(world, envmap)
    # robot side
    initial_pose = [0.0, -3.0, 0.0]
    logger_agent = LoggerAgent(0.2, 5.0 * pi / 180, dt, initial_pose)
    robot = RealRobot(initial_pose, logger_agent, PsiCamera(landmarks); color = "red")
    push!(world, robot)
    anim = @animate for i = 1:60
        t = dt * i
        annota = "t = $(round(t, sigdigits=3))[s]"
        z = observations(robot.sensor_, robot.pose_)
        v, ω = decision(logger_agent, z)
        p = draw(world, annota)
        state_transition(robot, v, ω, dt; move_noise = true, vel_bias_noise = true)
    end
    if GUI
        gif(anim, "ch09_logger2.gif", fps = 0.5)
    end
end

using Plots, Random, Distributions, LinearAlgebra, JuliaProbo
import Plots: Plot
gr();

abstract type AbstractEdge end
struct ObsEdge <: AbstractEdge
    t1::Int64
    t2::Int64
    x1::Vector{Float64}
    x2::Vector{Float64}
    z1::Vector{Float64}
    z2::Vector{Float64}
    ω_ul::Matrix{Float64}
    ω_ur::Matrix{Float64}
    ω_ll::Matrix{Float64}
    ω_lr::Matrix{Float64}
    xi_u::Vector{Float64}
    xi_l::Vector{Float64}
    function ObsEdge(
        t1::Int64,
        t2::Int64,
        z1_::Vector{Float64},
        z2_::Vector{Float64},
        xs::Dict{Int64,Vector{Float64}},
        sensor_noise_rate = [0.14, 0.05, 0.05],
    )
        z1_[1] == z2_[1] || error("Landmark Ids do not match")
        x1 = copy(xs[t1])
        x2 = copy(xs[t2])
        z1 = z1_[2:4]
        z2 = z2_[2:4]
        θ1 = x1[3]
        ϕ1 = z1[2]
        θ2 = x2[3]
        ϕ2 = z2[2]
        s1 = sin(θ1 + ϕ1)
        c1 = cos(θ1 + ϕ1)
        s2 = sin(θ2 + ϕ2)
        c2 = cos(θ2 + ϕ2)

        ê = [
            z2[1] * c2 - z1[1] * c1,
            z2[1] * s2 - z1[1] * s1,
            z2[2] - z2[3] - z1[2] + z1[3],
        ]
        ê = ê + x2 - x1
        while ê[3] >= pi
            ê[3] -= 2 * pi
        end
        while ê[3] < -pi
            ê[3] += 2 * pi
        end

        Q₁ = Diagonal([
            (z1[1] * sensor_noise_rate[1])^2,
            sensor_noise_rate[2]^2,
            sensor_noise_rate[3]^2,
        ])
        R₁ = -[c1 (-z1[1]*s1) 0; s1 (z1[1]*c1) 0; 0 1 -1]
        Q₂ = Diagonal([
            (z2[1] * sensor_noise_rate[1])^2,
            sensor_noise_rate[2]^2,
            sensor_noise_rate[3]^2,
        ])
        R₂ = [c2 (-z2[1]*s2) 0; s2 (z2[1]*c2) 0; 0 1 -1]
        Σ = R₁ * Q₁ * transpose(R₁) + R₂ * Q₂ * transpose(R₂)
        Ω = inv(Σ)
        B1 = -[1 0 -z1[1]*s1; 0 1 z1[1]*c1; 0 0 1]
        B2 = [1 0 -z2[1]*s2; 0 1 z2[1]*c2; 0 0 1]
        ω_ul = transpose(B1) * Ω * B1
        ω_ur = transpose(B1) * Ω * B2
        ω_ll = transpose(B2) * Ω * B1
        ω_lr = transpose(B2) * Ω * B2
        xi_u = -transpose(B1) * Ω * ê
        xi_l = -transpose(B2) * Ω * ê
        new(t1, t2, x1, x2, z1, z2, ω_ul, ω_ur, ω_ll, ω_lr, xi_u, xi_l)
    end
end

mutable struct MotionEdge <: AbstractEdge
    t1::Int64
    t2::Int64
    x̂₁::Vector{Float64}
    x̂₂::Vector{Float64}
    ω_ul::Matrix{Float64}
    ω_ur::Matrix{Float64}
    ω_ll::Matrix{Float64}
    ω_lr::Matrix{Float64}
    xi_u::Vector{Float64}
    xi_l::Vector{Float64}
    function MotionEdge(
        t1::Int64,
        t2::Int64,
        xs::Dict{Int64,Vector{Float64}},
        us::Dict{Int64,Vector{Float64}},
        δ::Float64,
        motion_noise_stds = Dict("vv" => 0.19, "vω" => 0.001, "ωv" => 0.13, "ωω" => 0.2),
    )
        x̂₁ = copy(xs[t1])
        x̂₂ = copy(xs[t2])
        v, ω = us[t2][1], us[t2][2]
        if abs(ω) < 1e-5
            ω = sign(ω) * 1e-5
        end

        M = matM(v, ω, δ, motion_noise_stds)
        A = matA(v, ω, δ, x̂₁[3])
        F = matF(v, ω, δ, x̂₁[3])

        Ω = inv(A * M * transpose(A) + Matrix(0.0001I, 3, 3))
        ω_ul = transpose(F) * Ω * F
        ω_ur = -transpose(F) * Ω
        ω_ll = -Ω * F
        ω_lr = Ω
        x2 = state_transition(x̂₁, v, ω, δ)
        xi_u = transpose(F) * Ω * (x̂₂ - x2)
        xi_l = -Ω * (x̂₂ - x2)
        new(t1, t2, x̂₁, x̂₂, ω_ul, ω_ur, ω_ll, ω_lr, xi_u, xi_l)
    end
end

struct MapEdge <: AbstractEdge
    x::Vector{Float64}
    z::Vector{Float64}
    m::Vector{Float64}
    Ω::Matrix{Float64}
    xi::Vector{Float64}
    function MapEdge(
        t::Int64,
        z::Vector{Float64},
        head_t::Int64,
        head_z::Vector{Float64},
        xs::Dict{Int64,Vector{Float64}},
        sensor_noise_rate = [0.14, 0.05, 0.05],
    )
        x = copy(xs[t])
        m =
            x + [
                z[1] * cos(x[3] + z[2]),
                z[1] * sin(x[3] + z[2]),
                -xs[head_t][3] + z[2] - head_z[2] - z[3] + head_z[3],
            ]
        while m[3] >= pi
            m[3] -= 2.0 * pi
        end
        while m[3] < -pi
            m[3] += 2.0 * pi
        end
        Q₁ = Diagonal([
            (z[1] * sensor_noise_rate[1])^2,
            sensor_noise_rate[2]^2,
            sensor_noise_rate[3]^2,
        ])
        s1 = sin(x[3] + z[2])
        c1 = cos(x[3] + z[2])
        R = [-c1 (z[1]*s1) 0; -s1 (-z[1]*c1) 0; 0 -1 1]
        Ω = inv(R * Q₁ * transpose(R))
        xi = Ω * m
        new(x, copy(z), m, Ω, xi)
    end
end

function make_landmarks(hat_xs, zlist_landmark)
    ms = Dict{Int64,Vector{Float64}}()
    for (landmark_id, zlist) in zlist_landmark
        edges = Vector{MapEdge}(undef, 0)
        head_z = zlist[1]
        for z in zlist
            # step, ID of landmark, d, ϕ, m
            push!(
                edges,
                MapEdge(
                    convert(Int64, z[1]),
                    z[3:5],
                    convert(Int64, head_z[1]),
                    head_z[3:5],
                    hat_xs,
                ),
            )
        end
        average = sum([e.m for e in edges], dims = 1) / length(edges)

        Ω = zeros(Float64, 3, 3)
        xi = zeros(Float64, 3)
        for e in edges
            Ω += e.Ω
            xi += e.xi
        end
        ms[landmark_id] = inv(Ω) * xi
    end
    return ms
end

function make_ax(xlim, ylim)
    p = plot(aspect_ratio = :equal, xlim = xlim, ylim = ylim, xlabel = "X", ylabel = "Y")
    return p
end


function make_edges(
    hat_xs::Dict{Int64,Vector{Float64}},
    zlist::Dict{Int64,Vector{Vector{Float64}}},
)
    landmark_keys_zlist = Dict{Int64,Vector{Vector{Float64}}}()

    for (step, zs) in zlist
        for i = 1:lastindex(zs)
            z = zs[i]
            landmark_id = convert(Int64, z[1])
            if !haskey(landmark_keys_zlist, landmark_id)
                landmark_keys_zlist[landmark_id] = Vector{Vector{Float64}}(undef, 0)
            end
            # step, ID of landmark, d, ϕ, m
            push!(landmark_keys_zlist[landmark_id], vcat([step * 1.0], z))
        end
    end

    edges = Vector{AbstractEdge}(undef, 0)

    for (landmark_id, lm_zlist) in landmark_keys_zlist
        for i = 1:lastindex(lm_zlist)
            for j = (i+1):lastindex(lm_zlist)
                z1 = lm_zlist[i]
                z2 = lm_zlist[j]
                t1 = convert(Int64, z1[1])
                t2 = convert(Int64, z2[1])
                push!(edges, ObsEdge(t1, t2, z1[2:5], z2[2:5], hat_xs))
            end
        end
    end

    return edges, landmark_keys_zlist
end

function add_edge!(edge::AbstractEdge, Ω::Matrix{Float64}, xi::Vector{Float64})
    f1, f2 = edge.t1 * 3 + 1, edge.t2 * 3 + 1
    t1, t2 = f1 + 2, f2 + 2
    Ω[f1:t1, f1:t1] += edge.ω_ul
    Ω[f1:t1, f2:t2] += edge.ω_ur
    Ω[f2:t2, f1:t1] += edge.ω_ll
    Ω[f2:t2, f2:t2] += edge.ω_lr
    xi[f1:t1] += edge.xi_u
    xi[f2:t2] += edge.xi_l
end

function draw_traj!(xs, p::Plot{T}) where {T}
    # xs is dictionary
    poses = [xs[s] for s = 0:length(xs)-1]
    p = scatter!(
        [e[1] for e in poses],
        [e[2] for e in poses],
        markershape = :circle,
        markersize = 1,
        color = "black",
    )
    p = plot!(
        [e[1] for e in poses],
        [e[2] for e in poses],
        linewidth = 0.5,
        color = "black",
    )
end

function draw_observations!(
    xs::Dict{Int64,Vector{Float64}},
    zlist::Dict{Int64,Vector{Vector{Float64}}},
    p::Plot{T},
) where {T}
    for s = 0:length(xs)-1
        if !haskey(zlist, s)
            continue
        end
        v = zlist[s]
        N = size(v)[1]
        for i = 1:N
            x, y, θ = xs[s][1], xs[s][2], xs[s][3]
            obsv = v[i]
            d, ϕ = obsv[2], obsv[3]
            mx = x + d * cos(θ + ϕ)
            my = y + d * sin(θ + ϕ)
            p = plot!([x, mx], [y, my], color = "pink", alpha = 0.5)
        end
    end
end

function draw_edges!(edges::Vector{ObsEdge}, p::Plot{T}) where {T}
    for edge in edges
        p = plot!(
            [edge.x1[1], edge.x2[1]],
            [edge.x1[2], edge.x2[2]],
            color = "red",
            alpha = 0.5,
        )
    end
end

function draw_all(
    xlim,
    ylim,
    xs::Dict{Int64,Vector{Float64}},
    zlist::Dict{Int64,Vector{Vector{Float64}}},
    p::Plot{T};
    landmarks = Dict{Int64,Vector{Float64}}(),
) where {T}
    p = make_ax(xlim, ylim)
    draw_observations!(xs, zlist, p)
    draw_traj!(xs, p)
    for (k, landmark) in landmarks
        p = scatter!(
            [landmark[1]],
            [landmark[2]],
            markersize = 10,
            markershape = :star,
            color = "blue",
        )
    end
    # draw_edges!(edges, p)
    return p
end

# hat_xs contains all the keys in 0:length(xs), but z does not (there are some time steps with no observation)
function read_data(fname = "log.txt")
    hat_xs = Dict{Int64,Vector{Float64}}()
    zlist = Dict{Int64,Vector{Vector{Float64}}}()
    delta = 0.0
    us = Dict{Int64,Vector{Float64}}()

    fd = open(fname, "r")
    lines = readlines(fd)
    for i = 1:length(lines)
        line = lines[i]
        tokens = split(line, ' ')
        if tokens[1] == "delta"
            delta = parse(Float64, tokens[2])
            continue
        end
        step = parse(Int64, tokens[2])
        if tokens[1] == "x"
            # x, y, θ
            hat_xs[step] = [
                parse(Float64, tokens[3]),
                parse(Float64, tokens[4]),
                parse(Float64, tokens[5]),
            ]
        elseif tokens[1] == "z"
            if !haskey(zlist, step)
                zlist[step] = Vector{Vector{Float64}}(undef, 0)
            end
            # ID of landmark, d, ϕ, m
            z = [
                parse(Float64, tokens[3]),
                parse(Float64, tokens[4]),
                parse(Float64, tokens[5]),
                parse(Float64, tokens[6]),
            ]
            push!(zlist[step], copy(z))
        elseif tokens[1] == "u"
            us[step] = [parse(Float64, tokens[3]), parse(Float64, tokens[4])]
        end
    end
    return hat_xs, zlist, us, delta
end

@testset "ch09_graphslam9" begin
    hat_xs, zlist, us, delta = read_data("log.txt")
    dim = length(hat_xs) * 3 # dimension of trajectory
    diff1 = 0.0
    diff2 = 0.0
    xlim = [-5.0, 5.0]
    ylim = [-5.0, 5.0]
    xpos = xlim[1] + (xlim[2] - xlim[1]) * 0.3
    ypos = ylim[2] - 0.05 * (ylim[2] - ylim[1])

    anim = @animate for cnt = 1:1000
        edges, zlist_landmark = make_edges(hat_xs, zlist)

        for i = 0:length(hat_xs)-2
            push!(edges, MotionEdge(i, i + 1, hat_xs, us, delta))
        end
        Ω = zeros(Float64, dim, dim)
        xi = zeros(Float64, dim)
        Ω[1:3, 1:3] += Matrix(10000000.0I, 3, 3)

        for edge in edges
            add_edge!(edge, Ω, xi)
        end

        Δxs = inv(Ω) * xi

        for i = 0:length(hat_xs)-1
            hat_xs[i] += Δxs[i*3+1:(i+1)*3]
        end

        diff2 = diff1
        diff1 = norm(Δxs)
        annota = "$(cnt)-th iter: ϵ = $(round(diff1, sigdigits=3))"

        landmarks = make_landmarks(hat_xs, zlist_landmark)
        p = make_ax(xlim, ylim)
        p = draw_all(xlim, ylim, hat_xs, zlist, p; landmarks = landmarks)
        p = annotate!(xpos, ypos, annota)
        plot(p, legend = nothing)
        if abs(diff2 - diff1) < 0.00005
            break
        end
    end
    if GUI
        gif(anim, "ch09_graphslam8.gif", fps = 1)
    end
end
