module JuliaProbo

import Plots: Plot, plot, plot!, scatter!, annotate!, quiver!, text
import Random: rand
import Distributions: Exponential, Normal, Uniform, MvNormal, pdf
import LinearAlgebra: Diagonal
import StatsBase: sample, Weights

export AbstractObject, AbstractSensor, AbstractAgent, AbstractEstimator
export draw
export Agent, EstimatorAgent, decision
export IdealRobot, RealRobot, state_transition
export IdealCamera, RealCamera, observations, observation_function
export Landmark, Map, World, push!
export Particle, Mcl, motion_update, copy

# `include` order does matter(needs to be topologically sorted based on the type definition dependency)
include("types.jl")
include("world.jl")
include("agent.jl")
include("robot.jl")
include("sensor.jl")
include("mcl_estimator.jl")

end # module
