struct Goal <: AbstractObject
    x::Float64
    y::Float64
    radius::Float64
    value::Float64
    function Goal(x::Float64, y::Float64, radius = 0.3, value = 0.0)
        new(x, y, radius, value)
    end
end

function inside(g::Goal, pose::Vector{Float64})
    return g.radius > hypot(g.x - pose[1], g.y - pose[2])
end

function draw(g::Goal, p::Plot{T}) where {T}
    p = plot!([g.x, g.x], [g.y, g.y + 0.6], color = "black", linewidth = 2.0)
    p = plot!(
        [g.x + 0.08],
        [g.y + 0.5],
        markershape = :rtriangle,
        markersize = 9,
        color = "red",
    )
end

struct Puddle <: AbstractObject
    lowerleft::Vector{Float64}
    upperright::Vector{Float64}
    depth::Float64
    function Puddle(lowerleft::Vector{Float64}, upperright::Vector{Float64}, depth::Float64)
        new(lowerleft, upperright, depth)
    end
end

function inside(puddle::Puddle, pose::Vector{Float64})
    x = pose[1]
    y = pose[2]
    xl, xu = puddle.lowerleft[1], puddle.upperright[1]
    yl, yu = puddle.lowerleft[2], puddle.upperright[2]
    return (xl <= x <= xu) && (yl <= y <= yu)
end

function draw(puddle::Puddle, p::Plot{T}) where {T}
    w = puddle.upperright[1] - puddle.lowerleft[1]
    h = puddle.upperright[2] - puddle.lowerleft[2]
    x = puddle.lowerleft[1]
    y = puddle.lowerleft[2]
    shape_x, shape_y = [x, x + w, x + w, x], [y, y, y + h, y + h]
    shapes = [(shape_x[i], shape_y[i]) for i = 1:4]
    push!(shapes, (shape_x[1], shape_y[1]))
    p = plot!(shapes, seriestype = :shape, fillcolor = :blue, opacity = 0.15)
end

mutable struct PuddleWorld <: AbstractWorld
    objects_::Vector{AbstractObject}
    puddles_::Vector{Puddle}
    robots_::Vector{AbstractRobot}
    goals_::Vector{Goal}
    xlim_::Vector{Float64}
    ylim_::Vector{Float64}
    function PuddleWorld(xlim::Vector{Float64}, ylim::Vector{Float64})
        new(
            Vector{AbstractObject}(undef, 0),
            Vector{Puddle}(undef, 0),
            Vector{AbstractRobot}(undef, 0),
            Vector{Goal}(undef, 0),
            [xlim[1], xlim[2]],
            [ylim[1], ylim[2]],
        )
    end
end

function Base.push!(world::PuddleWorld, obj::AbstractObject)
    push!(world.objects_, obj)
    if obj isa AbstractRobot
        push!(world.robots_, obj)
    elseif obj isa Puddle
        push!(world.puddles_, obj)
    elseif obj isa Goal
        push!(world.goals_, obj)
    end
end

function puddle_depth(world::PuddleWorld, pose::Vector{Float64})
    return sum([p.depth * convert(Float64, inside(p, pose)) for p in world.puddles_])
end

function update_status(world::PuddleWorld)
    for iter = 1:lastindex(world.robots_)
        robot = world.robots_[iter]
        # use the ground truth position
        robot.agent_.puddle_depth_ = puddle_depth(world, robot.pose_)
        for g in world.goals_
            if inside(g, robot.pose_)
                robot.agent_.in_goal_ = true
                robot.agent_.final_value_ = g.value
            end
        end
    end
end
