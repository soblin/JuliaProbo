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
    function Mcl(initial_pose::Vector{Float64}, num::Int64)
        new([Particle(initial_pose) for i in 1:num])
    end
end

function draw(mcl::Mcl, p::Plot{T}) where T
    xs = [p.pose_[1] for p = mcl.particles_]
    ys = [p.pose_[2] for p = mcl.particles_]
    vxs = [cos(p.pose_[3]) for p = mcl.particles_]
    vys = [sin(p.pose_[3]) for p = mcl.particles_]
    p = quiver!(xs, ys, quiver=(vxs, vys), color="blue", alpha=0.5)
end
