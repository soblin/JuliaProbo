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

@userplot MyEllipse

@recipe function f(
    c::MyEllipse;
    n_std = 1,
    n_ellipse_vertices = 100,
    alpha = 0.0,
    color = "blue",
)
    μ, S = _covellipse_args(c.args; n_std = n_std)

    θ = range(0, 2π; length = n_ellipse_vertices)
    A = S * [cos.(θ)'; sin.(θ)']
    @series begin
        fillalpha --> 0.0
        linecolor --> color
        Shape(μ[1] .+ A[1, :], μ[2] .+ A[2, :])
    end
end

function _covellipse_args(
    (μ, Σ)::Tuple{AbstractVector{<:Real},AbstractMatrix{<:Real}};
    n_std::Real,
)
    size(μ) == (2,) && size(Σ) == (2, 2) ||
        error("covellipse requires mean of length 2 and covariance of size 2×2.")
    λ, U = eigen(Σ)
    μ, n_std * U * diagm(.√λ)
end
_covellipse_args(args; n_std) = error(
    "Wrong inputs for covellipse: $(typeof.(args)). " *
    "Expected real-valued vector μ, real-valued matrix Σ.",
)
