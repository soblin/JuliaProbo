struct PoseUniform
    low::Vector{Float64}
    upp::Vector{Float64}
    uni::Uniform{Float64}
    function PoseUniform(xlim::Vector{Float64}, ylim::Vector{Float64})
        new([xlim[1], ylim[1], 0], [xlim[2], ylim[2], 2 * pi], Uniform())
    end
end

function uniform(mv::PoseUniform)
    c = mv.upp - mv.low
    x = [rand(mv.uni) for i = 1:3]
    return mv.low .+ (x .* c)
end
