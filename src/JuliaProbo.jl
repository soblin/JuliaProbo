module JuliaProbo

using Reexport
import RecipesBase: @recipe
@reexport using Plots
import Plots: Plot, plot, plot!, scatter!, annotate!, quiver!, text, @userplot, @series
import Random: rand
import Distributions: Exponential, Normal, Uniform, MvNormal, pdf, mean, cov
import LinearAlgebra: Diagonal, I, eigen, diagm
import StatsBase: sample, Weights
import StatsPlots: covellipse!
import StatsFuns: chisqinvcdf

export AbstractObject,
    AbstractWorld, AbstractSensor, AbstractRobot, AbstractAgent, AbstractEstimator
export draw
export PoseUniform, uniform
export Agent, EstimatorAgent, LoggerAgent, decision
export IdealRobot, RealRobot, state_transition
export IdealCamera, RealCamera, PsiCamera, observations, observation_function
export Landmark,
    EstimatedLandmark, Map, World, Goal, Puddle, PuddleWorld, push!, getindex, puddle_depth
export Particle,
    copy,
    Mcl,
    KalmanFilter,
    KldMcl,
    ResetMcl,
    AMcl,
    motion_update,
    matQ,
    matH,
    matM,
    matA,
    matF
export MapParticle, FastSlam1, FastSlam2

# `include` order does matter(needs to be topologically sorted based on the type definition dependency)
include("types.jl")
include("util.jl")
include("world.jl")
include("agent.jl")
include("robot.jl")
include("sensor.jl")
include("mcl_estimator.jl")
include("kf_estimator.jl")
include("fastslam.jl")

end # module
