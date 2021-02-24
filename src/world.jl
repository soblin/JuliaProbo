using JuliaProbo

struct Landmark <: AbstractObject
    pos::Vector{Float64}
    id::Int64
    function Landmark(pos::Vector{Float64}, id::Int64)
        new(pos, id)
    end
end

function draw(mark::Landmark, p::Plot{T}) where T
    p = scatter!([mark.pos[1]], [mark.pos[2]], markershape=:star, markersize=10, color="orange")
    p = annotate!(mark.pos[1] + 0.5, mark.pos[2] + 0.5, text("id: $(mark.id)", 10))
end

mutable struct Map <: AbstractObject
    landmarks_::Vector{Landmark}
    function Map()
        new(Vector{Landmark}[])
    end
end

function Base.push!(map::Map, landmark::Landmark)
    push!(map.landmarks_, landmark)
end

function Base.push!(map::Map, landmarks::Vector{Landmark})
    for landmark = landmarks
        push!(map.landmarks_, landmark)
    end
end

function draw(map::Map, p::Plot{T}) where T
    for landmark = map.landmarks_
        draw(landmark, p)
    end
end

mutable struct World
    objects_::Vector{AbstractObject}
    xlim_::Vector{Float64}
    ylim_::Vector{Float64}
    debug_::Bool
    function World(xlim::Vector{Float64}, ylim::Vector{Float64}, debug=false)
        new(Vector{AbstractObject}[], [xlim[1], xlim[2]], [ylim[1], ylim[2]], debug)
    end
end

function Base.push!(world::World, obj::AbstractObject)
    push!(world.objects_, obj)
end

function draw(world::World, annota::String)
    p = plot(aspect_ratio=:equal, xlim=world.xlim_, ylim=world.ylim_)
    xpos = (world.xlim_[1] + world.xlim_[2]) / 2.0
    ypos = world.ylim_[2] - 0.1 * (world.ylim_[2] - world.ylim_[1])
    if annota != nothing
        p = annotate!(xpos, ypos, annota)
    end
    for obj = world.objects_
        draw(obj, p)
    end
    return p
end

function draw(world::World, annota::Nothing)
    p = plot(aspect_ratio=:equal, xlim=world.xlim_, ylim=world.ylim_)
    xpos = (world.xlim_[1] + world.xlim_[2]) / 2.0
    ypos = world.ylim_[2] - 0.1 * (world.ylim_[2] - world.ylim_[1])
    for obj = world.objects_
        draw(obj, p)
    end
    return p
end
