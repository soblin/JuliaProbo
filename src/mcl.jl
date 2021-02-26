using JuliaProbo

mutable struct Particle
    pose_::Vector{Float64}
    function Particle(pose::Vector{Float64})
        @assert length(pose) == 3
        new(copy(pose))
    end
end

mutable struct Mcl <: AbstractEstimator
    particles_::Vector{Particle}
    function Mcl(pose::Vector{Float64}, num::Int64)
        new([Particle(pose) for i in 1:num])
    end
end

function draw(mcl::Mcl, p::Plot{T}) where T end
