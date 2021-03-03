module JuliaProbo

import Plots: Plot, plot, plot!, scatter!, annotate!, quiver!, text
import Random: rand
import Distributions: Exponential, Normal, Uniform, MvNormal, pdf, mean, cov
import LinearAlgebra: Diagonal, I
import StatsBase: sample, Weights
import StatsPlots: covellipse!
import StatsFuns: chisqinvcdf

export AbstractObject, AbstractSensor, AbstractAgent, AbstractEstimator
export draw
export Agent, EstimatorAgent, decision
export IdealRobot, RealRobot, state_transition, PoseUniform, uniform
export IdealCamera, RealCamera, observations, observation_function
export Landmark, Map, World, push!, getindex
export Particle, copy, Mcl, KalmanFilter, KdlMcl, motion_update

# `include` order does matter(needs to be topologically sorted based on the type definition dependency)
include("types.jl")
include("world.jl")
include("agent.jl")
include("robot.jl")
include("sensor.jl")
include("mcl_estimator.jl")
include("kf_estimator.jl")

end # module
