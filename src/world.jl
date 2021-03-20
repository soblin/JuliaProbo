struct Landmark <: AbstractLandmark
    pos_::Vector{Float64}
    id::Int64
    function Landmark(pos::Vector{Float64}, id::Int64)
        new(pos, id)
    end
end

function Base.copy(lm::Landmark)
    return Landmark(copy(lm.pos_), lm.id)
end

function draw(mark::Landmark, p::Plot{T}) where {T}
    p = scatter!(
        [mark.pos_[1]],
        [mark.pos_[2]],
        markershape = :star,
        markersize = 10,
        color = "orange",
    )
    p = annotate!(mark.pos_[1] + 0.5, mark.pos_[2] + 0.5, text("id: $(mark.id)", 10))
end

mutable struct Map <: AbstractObject
    landmarks_::Vector{AbstractLandmark}
    function Map()
        new(Vector{AbstractLandmark}[])
    end
end

function Base.copy(src::Map)
    dst = Map()
    push!(dst, copy(src.landmarks_))
    return dst
end

function Base.push!(map::Map, landmark::AbstractLandmark)
    push!(map.landmarks_, landmark)
end

function Base.push!(map::Map, landmarks::Vector{T}) where {T<:AbstractLandmark}
    for landmark in landmarks
        push!(map.landmarks_, landmark)
    end
end

function draw(map::Map, p::Plot{T}) where {T}
    for landmark in map.landmarks_
        draw(landmark, p)
    end
end

function Base.getindex(map::Map, index::Int)
    return map.landmarks_[index]
end

mutable struct World <: AbstractWorld
    objects_::Vector{AbstractObject}
    xlim_::Vector{Float64}
    ylim_::Vector{Float64}
    debug_::Bool
    function World(xlim::Vector{Float64}, ylim::Vector{Float64}, debug = false)
        new(Vector{AbstractObject}[], [xlim[1], xlim[2]], [ylim[1], ylim[2]], debug)
    end
end

function Base.push!(world::World, obj::AbstractObject)
    push!(world.objects_, obj)
end

function draw(world::AbstractWorld, annota::String)
    p = plot(aspect_ratio = :equal, xlim = world.xlim_, ylim = world.ylim_)
    xpos = world.xlim_[1] + (world.xlim_[2] - world.xlim_[1]) * 0.2
    ypos = world.ylim_[2] - 0.1 * (world.ylim_[2] - world.ylim_[1])
    if annota != nothing
        p = annotate!(xpos, ypos, annota)
    end
    for obj in world.objects_
        draw(obj, p)
    end
    return p
end

function draw(world::AbstractWorld, annota::Nothing)
    p = plot(aspect_ratio = :equal, xlim = world.xlim_, ylim = world.ylim_)
    xpos = (world.xlim_[1] + world.xlim_[2]) / 2.0
    ypos = world.ylim_[2] - 0.1 * (world.ylim_[2] - world.ylim_[1])
    for obj in world.objects_
        draw(obj, p)
    end
    return p
end

struct Goal <: AbstractObject
    x::Float64
    y::Float64
    radius::Float64
    function Goal(x::Float64, y::Float64, radius = 0.3)
        new(x, y, radius)
    end
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
    yl, yu = puddle.lowerleft[1], puddle.upperright[2]
    return (xl < x < xu) && (yl < y < yu)
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
