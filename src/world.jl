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

mutable struct World
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

function draw(world::World, annota::String)
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

function draw(world::World, annota::Nothing)
    p = plot(aspect_ratio = :equal, xlim = world.xlim_, ylim = world.ylim_)
    xpos = (world.xlim_[1] + world.xlim_[2]) / 2.0
    ypos = world.ylim_[2] - 0.1 * (world.ylim_[2] - world.ylim_[1])
    for obj in world.objects_
        draw(obj, p)
    end
    return p
end
